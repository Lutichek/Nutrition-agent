"""
Адаптивная норма: фактический расход энергии по динамике веса (ревью R9).

Формула Миффлина ошибается у конкретного человека на ±10-20%, и это
видно по весу: ест по плану, а вес стоит — расход ниже формулы. Оценка:

    расход = средняя калорийность за период − скорость изменения веса × 7700

Что считается съеденным (решение владельца продукта, «план + поправки»):
день с записью — по записи (собранный план, а после «съел не по плану» —
с поправкой); день без записи — по последней известной норме.

Каждое правило — про тихий отказ:

* **Мало данных — формула.** Меньше MIN_WEIGHINS взвешиваний или период
  короче MIN_SPAN_DAYS — оценки нет: вес за пару дней скачет на 1-2 кг
  водой, и «расход» из такого наклона был бы шумом с уверенными цифрами.
* **Скачки веса — медиана наклонов** (оценка Тейла — Сена), а не
  прямая наименьших квадратов: одно взвешивание после солёного ужина
  (+2 кг воды) двигает МНК-наклон заметно, медиану — почти нет.
* **Шаг ограничен**: не больше ±STEP от прежнего расхода за пересчёт
  и не чаще раза в RECALC_DAYS дней. Если человек молча отступает от
  плана, оценка смещается — шаг не даёт норме уехать за неделю.
* **Предохранители** (темп ≤1% массы, нижняя граница калорийности,
  потолок белка) — в targets.compute_targets: оценка подставляется
  вместо формулы на входе, и всё дальнейшее действует как обычно.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from statistics import mean, median

from targets import KCAL_PER_KG_FAT

MIN_WEIGHINS = 4
MIN_SPAN_DAYS = 14
WINDOW_DAYS = 28
STEP = 0.10
RECALC_DAYS = 7


@dataclass
class Estimate:
    tdee: float
    kg_per_week: float
    intake: float
    days: int
    weighins: int


def _theil_sen(points: list[tuple[int, float]]) -> float:
    """Медиана наклонов по всем парам точек (кг в день)."""
    slopes = [(w2 - w1) / (d2 - d1)
              for i, (d1, w1) in enumerate(points) for d2, w2 in points[i + 1:] if d2 != d1]
    return median(slopes)


def estimate(weighins: dict[str, float], intake: dict[str, float],
             fallback_kcal: float, today: date) -> Estimate | None:
    """Расход по последним WINDOW_DAYS дням — или None, если данных мало."""
    since = today - timedelta(days=WINDOW_DAYS)
    points = sorted((date.fromisoformat(d), float(kg)) for d, kg in weighins.items()
                    if since <= date.fromisoformat(d) <= today)
    if len(points) < MIN_WEIGHINS:
        return None
    first, last = points[0][0], points[-1][0]
    span = (last - first).days
    if span < MIN_SPAN_DAYS:
        return None

    slope = _theil_sen([((d - first).days, kg) for d, kg in points])
    # Съеденное по дням периода: запись — как есть, пропуск — последняя
    # известная калорийность до него (норма, по которой человек жил).
    eaten = {date.fromisoformat(d): float(k) for d, k in intake.items()}
    known = [eaten[d] for d in sorted(eaten) if d < first]
    current = known[-1] if known else fallback_kcal
    per_day = []
    for offset in range(span):
        day = first + timedelta(days=offset)
        current = eaten.get(day, current)
        per_day.append(current)
    average = mean(per_day)
    return Estimate(tdee=average - slope * KCAL_PER_KG_FAT, kg_per_week=slope * 7,
                    intake=average, days=span, weighins=len(points))


def adapted_tdee(weighins: dict[str, float], intake: dict[str, float],
                 formula_tdee: float, fallback_kcal: float,
                 previous: dict | None, today: date) -> dict | None:
    """Расход для нормы: {"tdee", "at"} — или None, пока работает формула.

    previous — прошлое значение из сессии: пересчёт не чаще RECALC_DAYS,
    и сдвиг не больше ±STEP от него (или от формулы, если его не было).
    """
    if previous and (today - date.fromisoformat(previous["at"])).days < RECALC_DAYS:
        return previous
    found = estimate(weighins, intake, fallback_kcal, today)
    if found is None:
        return previous  # данные кончились — держим прошлую оценку, не прыгаем к формуле
    base = previous["tdee"] if previous else formula_tdee
    tdee = min(max(found.tdee, base * (1 - STEP)), base * (1 + STEP))
    return {"tdee": round(tdee), "at": today.isoformat()}
