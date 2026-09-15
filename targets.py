"""
Нормы КБЖУ по профилю пользователя: профиль → целевые числа на день.

Второй детерминированный слой проекта. Здесь формулы, а не мнение модели:
одинаковый профиль всегда даёт одинаковые цифры, и результат можно проверить
на бумаге. LLM в этом модуле не вызывается ни разу.

Почему так, а не «спросим у модели»: калорийность — это то, что пользователь
понесёт в жизнь. Ошибка на 400 ккал в день — это ошибка на два килограмма
в месяц. Формулу можно проверить, галлюцинацию — нет.

Что считается:

1. **BMR** — базовый обмен по уравнению Миффлина–Сан Жеора. Выбрано потому,
   что на современных выборках оно точнее Харриса–Бенедикта.
2. **TDEE** — BMR × коэффициент активности.
3. **Целевая калорийность** — TDEE со сдвигом под цель (дефицит или профицит).
4. **Макронутриенты** — белок от массы тела, жиры от калорийности, углеводы
   добирают остаток.

Отдельно — предохранители (``_apply_safety_floor``): цель не может опуститься
ниже физиологического минимума, а темп снижения веса ограничен сверху.
Агент не имеет права их обойти.

⚠️ Расчёт носит справочный характер и не заменяет консультацию врача
или диетолога.

Использование::

    from targets import Profile, compute_targets

    profile = Profile(sex="female", age=31, height_cm=168, weight_kg=74, activity="light")
    targets = compute_targets(profile)
    print(targets.kcal, targets.protein_g)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# ────────────────────────────────────────────────────────────
# Константы
# ────────────────────────────────────────────────────────────

# Коэффициенты физической активности (PAL) к базовому обмену.
ACTIVITY_FACTORS: dict[str, float] = {
    "sedentary": 1.2,   # сидячая работа, тренировок нет
    "light": 1.375,     # тренировки 1 раз в неделю
    "moderate": 1.55,   # тренировки 2-3 раза в неделю
    "high": 1.725,      # тренировки 4 раза в неделю и больше
    "athlete": 1.9,     # две тренировки в день или тяжёлый физический труд
}

# Те же уровни словами человека. Нужны в двух местах: когда агент
# спрашивает про активность и когда объясняет, откуда взялась норма.
#
# Формулировки бытовые намеренно. «Moderate» человеку ничего не говорит,
# а разница между уровнями — это 30% суточной калорийности: у мужчины 65 кг
# между «сидячим» и «высоким» лежит 340 ккал в день.
# Цели по-русски. Держатся рядом с самими целями: три копии этого словаря
# в agent.py, export.py и здесь уже разъехались — добавление «recomp»
# уронило ответ с KeyError, потому что обновили только один из них.
GOAL_LABELS: dict[str, str] = {
    "lose": "снижение веса",
    "maintain": "удержание веса",
    "gain": "набор массы",
    "recomp": "рекомпозиция тела",
}

ACTIVITY_LABELS: dict[str, str] = {
    "sedentary": "сидячий образ жизни — сидячая работа, тренировок нет",
    "light": "малая активность — тренировки 1 раз в неделю",
    "moderate": "средняя активность — тренировки 2-3 раза в неделю",
    "high": "высокая активность — тренировки 4 раза в неделю и больше",
    "athlete": "спортсмен — две тренировки в день или тяжёлый физический труд",
}

# Один килограмм жировой ткани — примерно 7700 ккал. Отсюда пересчёт
# «сколько килограммов в неделю» в «сколько ккал в день».
KCAL_PER_KG_FAT = 7700.0

# Предохранитель: ниже этой калорийности рацион не покрывает потребность
# в микронутриентах без медицинского наблюдения.
MIN_KCAL: dict[str, float] = {"female": 1200.0, "male": 1500.0}

# Предохранитель: безопасный темп снижения веса — не больше 1% массы тела
# в неделю; сверху ограничиваем ещё и абсолютом.
MAX_WEEKLY_LOSS_FRACTION = 0.01
MAX_WEEKLY_LOSS_KG = 1.0

# Белок, г на кг массы тела. При дефиците калорий его нужно больше:
# он удерживает мышечную массу, которая иначе уходит вместе с жиром.
#
# «recomp» — рекомпозиция: одновременно снизить жир и нарастить мышцы.
# Калорийность при этом поддерживающая, как у «maintain», а белка нужно
# столько же, сколько при наборе, — именно он и делает всю работу.
# Ставить рекомпозиции 1.4 г/кг, как обычному поддержанию, бессмысленно:
# без белка на нулевом балансе калорий мышцы не растут.
PROTEIN_PER_KG: dict[str, float] = {
    "lose": 1.8, "maintain": 1.4, "gain": 1.8, "recomp": 2.0,
}

# Потолок доли белка в калорийности. Без него формула «г на кг массы тела»
# на низкой калорийности даёт абсурд: женщине 74 кг при 1219 ккал выходило
# 133 г белка — 44% калорийности, столько просто не съесть. А мужчине 200 кг
# при 2656 ккал — 360 г, то есть 54%.
#
# Причина в том, что потребность в белке масштабируется массой тела,
# а калорийность на дефиците — нет. У людей с большим лишним весом эти две
# величины расходятся, потому что жировая ткань белка не требует.
# 35% — верхняя граница, за которой рацион перестаёт быть исполнимым.
MAX_PROTEIN_KCAL_SHARE = 0.35

# Нижняя граница нормы белка: меньше нельзя даже при жёстком дефиците,
# иначе снижение веса идёт за счёт мышц.
MIN_PROTEIN_PER_KG = 1.2

# Доля калорийности из жиров. Ниже 20% страдает усвоение жирорастворимых
# витаминов и синтез гормонов, поэтому это тоже предохранитель.
FAT_KCAL_SHARE = 0.28
MIN_FAT_KCAL_SHARE = 0.20

# Клетчатка: нормируется на калорийность, а не на массу тела.
FIBER_G_PER_1000_KCAL = 14.0

KCAL_PER_G_PROTEIN = 4.0
KCAL_PER_G_FAT = 9.0
KCAL_PER_G_CARB = 4.0

Sex = Literal["female", "male"]
Activity = Literal["sedentary", "light", "moderate", "high", "athlete"]
Goal = Literal["lose", "maintain", "gain", "recomp"]


# ────────────────────────────────────────────────────────────
# Модели
# ────────────────────────────────────────────────────────────


class Profile(BaseModel):
    """Профиль пользователя — вход для расчёта норм."""

    sex: Sex
    age: int = Field(ge=14, le=100, description="Полных лет")
    height_cm: float = Field(ge=120, le=230)
    weight_kg: float = Field(ge=35, le=250)
    activity: Activity = "sedentary"
    goal: Goal = "maintain"

    # Желаемый темп изменения веса, кг в неделю. Если не задан — берём
    # умеренный по умолчанию (см. _default_rate).
    rate_kg_per_week: float | None = Field(default=None, ge=0.0, le=2.0)

    # Что человек не ест: аллергии, убеждения, непереносимость.
    # Список подстрок, которые не должны встречаться в названии блюда.
    exclude: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_rate_matches_goal(self) -> Profile:
        if self.goal == "maintain" and self.rate_kg_per_week:
            raise ValueError("При цели maintain темп изменения веса должен быть нулевым")
        return self


class Targets(BaseModel):
    """Целевые показатели на день — выход расчёта."""

    kcal: float
    protein_g: float
    fat_g: float
    carb_g: float
    fiber_g: float

    # Служебные величины: нужны, чтобы объяснить пользователю, откуда цифры.
    bmr: float
    tdee: float
    goal: Goal
    rate_kg_per_week: float

    # Что сработало из предохранителей — агент обязан это проговорить.
    adjustments: list[str] = Field(default_factory=list)

    def as_dict(self) -> dict[str, float]:
        """Только числа рациона — удобно для сравнения с собранным днём."""
        return {
            "kcal": self.kcal,
            "protein_g": self.protein_g,
            "fat_g": self.fat_g,
            "carb_g": self.carb_g,
            "fiber_g": self.fiber_g,
        }


# ────────────────────────────────────────────────────────────
# Формулы
# ────────────────────────────────────────────────────────────


def mifflin_st_jeor(profile: Profile) -> float:
    """Базовый обмен (ккал/сутки) по уравнению Миффлина–Сан Жеора."""
    base = 10 * profile.weight_kg + 6.25 * profile.height_cm - 5 * profile.age
    return base + (5 if profile.sex == "male" else -161)


def total_expenditure(profile: Profile) -> float:
    """Суточный расход с поправкой на активность."""
    return mifflin_st_jeor(profile) * ACTIVITY_FACTORS[profile.activity]


def _default_rate(goal: Goal) -> float:
    """Темп по умолчанию, если пользователь его не назвал."""
    if goal == "lose":
        return 0.5   # умеренный дефицит: держится дольше агрессивного
    if goal == "gain":
        return 0.25  # быстрее — значит больше жира, а не мышц
    return 0.0       # maintain и recomp: вес держим, меняется состав тела


def _cap_rate(profile: Profile, rate: float, adjustments: list[str]) -> float:
    """Ограничить темп снижения веса безопасным пределом."""
    if profile.goal != "lose":
        return rate

    limit = min(MAX_WEEKLY_LOSS_KG, profile.weight_kg * MAX_WEEKLY_LOSS_FRACTION)
    if rate > limit:
        adjustments.append(
            f"Темп снижен с {rate:.2f} до {limit:.2f} кг/нед: быстрее — это потеря "
            f"мышечной массы, а не жира (предел — 1% массы тела в неделю)"
        )
        return limit
    return rate


def _apply_safety_floor(profile: Profile, kcal: float, adjustments: list[str]) -> float:
    """Не дать целевой калорийности упасть ниже физиологического минимума."""
    floor = MIN_KCAL[profile.sex]
    if kcal < floor:
        adjustments.append(
            f"Калорийность поднята с {kcal:.0f} до {floor:.0f} ккал: ниже этого "
            f"рацион не покрывает потребность в витаминах и минералах"
        )
        return floor
    return kcal


def compute_targets(profile: Profile) -> Targets:
    """Посчитать целевые КБЖУ на день по профилю.

    Порядок важен: сначала считаем «чего хочет пользователь», затем прогоняем
    результат через предохранители и записываем каждое срабатывание
    в ``adjustments`` — чтобы агент не выдал урезанную цифру молча.
    """
    adjustments: list[str] = []

    bmr = mifflin_st_jeor(profile)
    tdee = total_expenditure(profile)

    rate = profile.rate_kg_per_week
    if rate is None:
        rate = _default_rate(profile.goal)
    rate = _cap_rate(profile, rate, adjustments)

    # Сдвиг калорийности под цель.
    daily_shift = rate * KCAL_PER_KG_FAT / 7.0
    if profile.goal == "lose":
        kcal = tdee - daily_shift
    elif profile.goal == "gain":
        kcal = tdee + daily_shift
    else:
        kcal = tdee

    kcal = _apply_safety_floor(profile, kcal, adjustments)

    # ── Макронутриенты ───────────────────────────────────────
    # Белок считаем от массы тела, а не от калорийности: при дефиците
    # потребность в белке не падает вместе с калориями.
    protein_g = PROTEIN_PER_KG[profile.goal] * profile.weight_kg

    # ...но ограничиваем сверху долей калорийности, иначе на низкой цели
    # получается рацион, который физически не съесть (см. MAX_PROTEIN_KCAL_SHARE).
    protein_ceiling_g = kcal * MAX_PROTEIN_KCAL_SHARE / KCAL_PER_G_PROTEIN
    if protein_g > protein_ceiling_g:
        adjustments.append(
            f"Норма белка снижена с {protein_g:.0f} до {protein_ceiling_g:.0f} г: "
            f"иначе белок занимал бы больше {MAX_PROTEIN_KCAL_SHARE:.0%} калорийности, "
            f"а столько за день не съесть"
        )
        protein_g = protein_ceiling_g

    protein_kcal = protein_g * KCAL_PER_G_PROTEIN

    fat_g = kcal * FAT_KCAL_SHARE / KCAL_PER_G_FAT
    fat_kcal = fat_g * KCAL_PER_G_FAT

    carb_kcal = kcal - protein_kcal - fat_kcal

    # Крайний случай: при большом дефиците и высокой массе тела белок с жирами
    # съедают всю калорийность. Тогда ужимаем жиры до нижней границы, а если
    # и этого мало — режем белок, но не ниже 1.2 г/кг.
    if carb_kcal < 0:
        fat_kcal = kcal * MIN_FAT_KCAL_SHARE
        fat_g = fat_kcal / KCAL_PER_G_FAT
        carb_kcal = kcal - protein_kcal - fat_kcal
        adjustments.append("Доля жиров снижена до 20% калорийности: иначе не остаётся углеводов")

    if carb_kcal < 0:
        protein_g = MIN_PROTEIN_PER_KG * profile.weight_kg
        protein_kcal = protein_g * KCAL_PER_G_PROTEIN
        carb_kcal = max(kcal - protein_kcal - fat_kcal, 0.0)
        adjustments.append(
            f"Норма белка снижена до {MIN_PROTEIN_PER_KG} г/кг: при такой "
            f"калорийности полноценный рацион иначе не собирается"
        )

    return Targets(
        kcal=round(kcal),
        protein_g=round(protein_g),
        fat_g=round(fat_g),
        carb_g=round(carb_kcal / KCAL_PER_G_CARB),
        fiber_g=round(kcal / 1000 * FIBER_G_PER_1000_KCAL),
        bmr=round(bmr),
        tdee=round(tdee),
        goal=profile.goal,
        rate_kg_per_week=rate,
        adjustments=adjustments,
    )


def explain(profile: Profile, targets: Targets) -> str:
    """Человекочитаемое объяснение расчёта — для ответа агента."""
    lines = [
        f"Базовый обмен (Миффлин–Сан Жеор): {targets.bmr:.0f} ккал",
        f"С учётом активности — {ACTIVITY_LABELS[profile.activity]} "
        f"(×{ACTIVITY_FACTORS[profile.activity]}): {targets.tdee:.0f} ккал",
    ]

    if profile.goal == "lose":
        lines.append(f"Цель — снижение веса на {targets.rate_kg_per_week:.2f} кг/нед")
    elif profile.goal == "gain":
        lines.append(f"Цель — набор массы {targets.rate_kg_per_week:.2f} кг/нед")
    elif profile.goal == "recomp":
        lines.append(
            "Цель — рекомпозиция: вес держим, состав тела меняем. "
            "Калорийность поддерживающая, а белка больше обычного — "
            "именно он растит мышцы на нулевом балансе калорий"
        )
    else:
        lines.append("Цель — удержание веса")

    lines.append(
        f"Норма на день: {targets.kcal:.0f} ккал, "
        f"белки {targets.protein_g:.0f} г, жиры {targets.fat_g:.0f} г, "
        f"углеводы {targets.carb_g:.0f} г, клетчатка {targets.fiber_g:.0f} г"
    )

    for note in targets.adjustments:
        lines.append(f"⚠️ {note}")

    return "\n".join(lines)


def main() -> None:
    """Показать расчёт на нескольких профилях — быстрая проверка формул."""
    examples = [
        Profile(sex="female", age=31, height_cm=168, weight_kg=74, activity="light", goal="lose"),
        Profile(sex="male", age=40, height_cm=182, weight_kg=95, activity="moderate", goal="lose",
                rate_kg_per_week=1.5),
        Profile(sex="male", age=25, height_cm=178, weight_kg=63, activity="high", goal="gain"),
        Profile(sex="female", age=55, height_cm=160, weight_kg=58, activity="sedentary"),
    ]

    for profile in examples:
        print("=" * 70)
        print(f"{profile.sex}, {profile.age} лет, {profile.height_cm:.0f} см, "
              f"{profile.weight_kg:.0f} кг, активность {profile.activity}, цель {profile.goal}")
        print(explain(profile, compute_targets(profile)))
        print()


if __name__ == "__main__":
    main()


# ────────────────────────────────────────────────────────────
# Циклирование по дням недели
# ────────────────────────────────────────────────────────────
# Одна и та же калорийность каждый день — упрощение. В дни тренировок
# организму нужно больше углеводов и энергии, в дни отдыха меньше.
# Практика силовых видов спорта давно это разводит, и человек, который
# тренируется, ждёт именно такого плана.
#
# Что меняется, а что нет:
#
#   белок  — НЕ меняется. Он считается от массы тела, а масса тела
#            в выходной та же. Потребность в белке не зависит от того,
#            была ли тренировка.
#   жир    — НЕ меняется. Он отвечает за гормоны и усвоение витаминов,
#            это фон, а не топливо под нагрузку.
#   углеводы — забирают всю разницу. Они и есть топливо.
#
# Недельная сумма при этом РАВНА обычной норме: циклирование
# перераспределяет калории, а не добавляет дефицит поверх цели.
# Дефицит или профицит уже заложен в саму норму через goal, и удваивать
# его нельзя — получилось бы совсем не то, что человек просил.

# Сколько тренировок в неделю подразумевает каждый уровень активности.
# Значения из ACTIVITY_LABELS: там человеку показаны те же числа.
TRAINING_DAYS: dict[str, int] = {
    "sedentary": 0,
    "light": 1,
    "moderate": 3,
    "high": 4,
    "athlete": 6,
}

# Насколько тренировочный день калорийнее обычной нормы.
#
# 10% — умеренная надбавка: заметная, но не превращающая тренировочный день
# в праздник. Дальше растёт дефицит выходного дня, а он ограничен снизу
# теми же предохранителями, что и обычная норма.
TRAINING_DAY_UPLIFT = 0.10

# Ниже этой доли калорий из углеводов день перестаёт быть съедобным:
# белок и жир фиксированы, и всё, что остаётся, — это углеводы.
MIN_CARB_KCAL_SHARE = 0.15

# Насколько глубоко может просесть день отдыха относительно нормы.
#
# Предохранитель против арифметики: чем больше тренировок, тем меньше
# выходных, на которые ложится вся компенсация. У спортсмена с шестью
# тренировками единственный выходной выходил на 40% нормы — 1200 ккал
# и 45 г углеводов вместо 3000. Формально среднее сходилось, практически
# это голодный день.
#
# Поэтому просадка ограничена, а надбавка тренировочного дня подстраивается
# под неё: лучше меньше разводить дни, чем выдать несъедобный выходной.
MIN_REST_DAY_FACTOR = 0.85


class CycledTargets(BaseModel):
    """Нормы на тренировочный день и на день отдыха.

    ``training_days`` — сколько тренировок в неделю, ``rest_days`` — сколько
    выходных. Недельное среднее по калорийности равно ``base.kcal``.
    """

    base: Targets
    training: Targets
    rest: Targets
    training_days: int
    rest_days: int

    @property
    def weekly_mean_kcal(self) -> float:
        """Среднее за неделю — обязано совпадать с базовой нормой."""
        total = self.training.kcal * self.training_days + self.rest.kcal * self.rest_days
        return total / (self.training_days + self.rest_days)

    def for_day(self, index: int) -> Targets:
        """Норма на день по его номеру от нуля.

        Тренировки ставятся первыми днями недели и повторяются каждые семь
        дней. Это упрощение: настоящее расписание у всех своё, а спрашивать
        его — ещё один вопрос перед расчётом.
        """
        return self.training if index % 7 < self.training_days else self.rest


def _shift_carbs(base: Targets, kcal: float, note: str) -> Targets:
    """Пересобрать норму под другую калорийность, двигая только углеводы."""
    protein_kcal = base.protein_g * KCAL_PER_G_PROTEIN
    fat_kcal = base.fat_g * KCAL_PER_G_FAT
    carb_kcal = kcal - protein_kcal - fat_kcal

    adjustments = list(base.adjustments)
    fat_g = base.fat_g

    # На выходном дне углеводов может не остаться вовсе: белок и жир
    # фиксированы, а калорийность упала. Тогда ужимаем жир — но не ниже
    # его собственного предохранителя.
    if carb_kcal < kcal * MIN_CARB_KCAL_SHARE:
        fat_kcal = max(kcal * MIN_FAT_KCAL_SHARE, kcal - protein_kcal - kcal * MIN_CARB_KCAL_SHARE)
        fat_g = fat_kcal / KCAL_PER_G_FAT
        carb_kcal = kcal - protein_kcal - fat_kcal
        adjustments.append(
            f"В день отдыха жиры снижены до {fat_g:.0f} г: иначе на углеводы "
            f"не осталось бы места"
        )

    return base.model_copy(update={
        "kcal": round(kcal),
        "fat_g": round(fat_g),
        "carb_g": round(max(carb_kcal, 0) / KCAL_PER_G_CARB),
        "fiber_g": round(kcal / 1000 * FIBER_G_PER_1000_KCAL),
        "adjustments": adjustments + [note],
    })


def cycle_targets(profile: Profile, base: Targets | None = None) -> CycledTargets | None:
    """Развести норму на тренировочные дни и дни отдыха.

    Возвращает ``None``, если человек не тренируется: разводить нечего,
    и показывать две одинаковые карточки было бы обманом.

    Args:
        profile: профиль человека — из него берётся частота тренировок.
        base: обычная норма; если не передана, считается здесь же.
    """
    base = base or compute_targets(profile)
    training_days = TRAINING_DAYS.get(profile.activity, 0)

    if not 1 <= training_days <= 6:
        return None

    rest_days = 7 - training_days

    # Калорийность тренировочного дня задаём надбавкой, выходного —
    # выводим из условия «недельное среднее равно норме». Так дефицит
    # или профицит цели остаётся ровно таким, каким его посчитали.
    #
    # Надбавку приходится ограничивать: при шести тренировках вся
    # компенсация ложится на один выходной, и он проваливается. Считаем
    # максимум, при котором выходной ещё не уходит ниже MIN_REST_DAY_FACTOR,
    # и берём меньшее из двух.
    max_uplift = (7 - MIN_REST_DAY_FACTOR * rest_days) / training_days - 1
    uplift = min(TRAINING_DAY_UPLIFT, max(max_uplift, 0.0))

    training_kcal = base.kcal * (1 + uplift)
    rest_kcal = (base.kcal * 7 - training_kcal * training_days) / rest_days

    return CycledTargets(
        base=base,
        training=_shift_carbs(
            base, training_kcal,
            f"Тренировочный день: +{uplift:.0%} калорий, разница в углеводах"),
        rest=_shift_carbs(base, rest_kcal,
                          "День отдыха: калорийность ниже, недельное среднее — норма"),
        training_days=training_days,
        rest_days=rest_days,
    )
