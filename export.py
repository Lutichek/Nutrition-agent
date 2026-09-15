"""
Выгрузка меню в Excel.

Текстовый файл годится, чтобы прочитать меню один раз, и ни на что больше:
в нём нельзя отфильтровать дни, посчитать сумму по своей колонке или увидеть,
куда уезжает рацион за месяц. Человек, который получил меню на 30 дней,
почти наверняка захочет с ним что-то сделать — отсюда таблица.

Книга состоит из трёх листов:

``Сводка``
    Норма, среднесуточный факт, отклонение и три графика: из чего набираются
    калории, насколько факт отходит от нормы и как калорийность гуляет по дням.

``Меню``
    Плоская таблица «одна строка — один приём пищи» с автофильтром.
    Плоская намеренно: по такой таблице работают сводные, сортировка
    и фильтр, а по красиво свёрстанной — нет.

``По дням``
    Итоги каждого дня. Отсюда же берут данные графики со «Сводки»:
    в Excel график всегда ссылается на ячейки, а не хранит числа внутри.

Модуль ничего не знает про HTTP и про агента — на вход ``Menu``, на выходе
книга или готовые байты. Это делает его пригодным и для API, и для терминала,
и для тестов.
"""

from __future__ import annotations

import io
from typing import Any

from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, PieChart, Reference
from openpyxl.chart.label import DataLabelList
from openpyxl.chart.marker import DataPoint
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.worksheet import Worksheet

from restrictions import describe as describe_restrictions
from solver import SODIUM_LIMIT_MG, Menu
from targets import GOAL_LABELS

# ── Палитра ──────────────────────────────────────────────────
# Три слота категориальной палитры, проверенные на различимость при
# дальтонизме (худшая пара ΔE 9.2 при пороге 8) и на контраст с белым фоном.
# Порядок закреплён: белки всегда синие, жиры оранжевые, углеводы бирюзовые —
# на всех графиках книги, иначе читателю пришлось бы каждый раз смотреть
# в легенду.
PROTEIN = "2A78D6"
FAT = "EB6834"
CARB = "1BAF7A"

# Бирюзовый на белом даёт контраст 2.74:1 — ниже порога 3:1. Поэтому везде,
# где он используется, подписи вынесены прямо на график, а не в одну легенду.

INK = "16191D"
MUTED = "6B7280"
LINE = "D8DCE0"
HEAD_BG = "EEF2F7"

# ── Оформление ───────────────────────────────────────────────
TITLE_FONT = Font(size=14, bold=True, color=INK)
HEAD_FONT = Font(size=10, bold=True, color=INK)
MUTED_FONT = Font(size=10, color=MUTED)
BOLD = Font(bold=True, color=INK)

HEAD_FILL = PatternFill("solid", fgColor=HEAD_BG)
THIN = Side(style="thin", color=LINE)
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

# Калорийность макронутриентов — коэффициенты Этуотера. Нужны, чтобы
# показать не граммы, а доли калорий: 60 г жира и 60 г белка выглядят
# одинаково, хотя жир даёт вдвое больше энергии.
KCAL_PER_GRAM = {"protein_g": 4.0, "fat_g": 9.0, "carb_g": 4.0}

MACROS = [("Белки", "protein_g", PROTEIN),
          ("Жиры", "fat_g", FAT),
          ("Углеводы", "carb_g", CARB)]

def _fit_columns(sheet: Worksheet, widths: dict[str, int]) -> None:
    for letter, width in widths.items():
        sheet.column_dimensions[letter].width = width


def _header_row(sheet: Worksheet, row: int, titles: list[str]) -> None:
    for column, title in enumerate(titles, start=1):
        cell = sheet.cell(row=row, column=column, value=title)
        cell.font = HEAD_FONT
        cell.fill = HEAD_FILL
        cell.border = BOX
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    sheet.row_dimensions[row].height = 28


def _averages(menu: Menu) -> dict[str, float]:
    """Среднесуточные итоги по всему меню."""
    keys = ["kcal", "protein_g", "fat_g", "carb_g", "fiber_g", "sodium_mg"]
    count = len(menu.days) or 1
    return {
        key: sum(day.totals.get(key, 0.0) for day in menu.days) / count
        for key in keys
    }


# ────────────────────────────────────────────────────────────
# Лист «По дням»
# ────────────────────────────────────────────────────────────


def _sheet_by_day(book: Workbook, menu: Menu) -> Worksheet:
    sheet = book.create_sheet("По дням")

    # «Норма» стоит второй колонкой не случайно: график ниже берёт
    # диапазон B:C, а соседние колонки — единственный способ отдать
    # Excel две серии одним диапазоном.
    _header_row(sheet, 1, ["День", "Калории", "Норма", "Белки, г", "Жиры, г",
                           "Углеводы, г", "Клетчатка, г", "Натрий, мг",
                           "Калории к норме"])

    target_kcal = menu.targets["kcal"] or 1

    for number, day in enumerate(menu.days, start=1):
        row = number + 1
        totals = day.totals
        values = [
            number,
            round(totals["kcal"]),
            round(target_kcal),
            round(totals["protein_g"]),
            round(totals["fat_g"]),
            round(totals["carb_g"]),
            round(totals.get("fiber_g", 0)),
            round(totals.get("sodium_mg", 0)),
            totals["kcal"] / target_kcal,
        ]
        for column, value in enumerate(values, start=1):
            cell = sheet.cell(row=row, column=column, value=value)
            cell.border = BOX
            cell.number_format = "0%" if column == 9 else "0"

    _fit_columns(sheet, {"A": 7, "B": 10, "C": 10, "D": 11, "E": 10,
                         "F": 13, "G": 14, "H": 12, "I": 16})
    sheet.freeze_panes = "A2"
    return sheet


# ────────────────────────────────────────────────────────────
# Лист «Меню»
# ────────────────────────────────────────────────────────────


def _sheet_menu(book: Workbook, menu: Menu) -> Worksheet:
    sheet = book.create_sheet("Меню")

    _header_row(sheet, 1, ["День", "Приём пищи", "Блюдо", "Граммы", "Калории",
                           "Белки, г", "Жиры, г", "Углеводы, г",
                           "Клетчатка, г", "Натрий, мг"])

    row = 2
    for number, day in enumerate(menu.days, start=1):
        for item in day.items:
            values = [
                number, item.slot, item.title,
                round(item.grams), round(item.kcal),
                round(item.protein_g), round(item.fat_g), round(item.carb_g),
                round(item.fiber_g), round(item.sodium_mg),
            ]
            for column, value in enumerate(values, start=1):
                cell = sheet.cell(row=row, column=column, value=value)
                cell.border = BOX
                if column >= 4:
                    cell.number_format = "0"
            row += 1

    # Автофильтр и закреплённая шапка: на 30 днях это 210 строк, и без них
    # таблица бесполезна — до конца не долистать, а шапка уезжает.
    sheet.auto_filter.ref = f"A1:J{row - 1}"
    sheet.freeze_panes = "A2"

    _fit_columns(sheet, {"A": 7, "B": 18, "C": 52, "D": 9, "E": 10,
                         "F": 11, "G": 10, "H": 13, "I": 14, "J": 12})
    return sheet


# ────────────────────────────────────────────────────────────
# Графики
# ────────────────────────────────────────────────────────────


def _plain_labels(chart) -> None:
    """Подписи прямо на данных: только значение, без имени серии.

    По умолчанию Excel склеивает в подпись всё подряд и выдаёт
    «Калорий в день, Белки, 35%». Читать это невозможно.
    """
    chart.dataLabels = DataLabelList()
    chart.dataLabels.showSerName = False
    chart.dataLabels.showCatName = False
    chart.dataLabels.showLegendKey = False
    chart.dataLabels.showBubbleSize = False


def _chart_macro_split(sheet: Worksheet, first_row: int) -> PieChart:
    """Из чего набираются калории: доли белков, жиров и углеводов.

    Круговая диаграмма здесь уместна ровно потому, что случай для неё
    и создан: три доли одного целого, в сумме сто процентов.
    """
    chart = PieChart()
    chart.title = "Откуда берутся калории"
    chart.height, chart.width = 8.0, 11.0

    labels = Reference(sheet, min_col=1, min_row=first_row, max_row=first_row + 2)
    data = Reference(sheet, min_col=2, min_row=first_row - 1, max_row=first_row + 2)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(labels)

    # Название и процент пишутся прямо на секторах, легенда убрана.
    # Подписи здесь не украшение: бирюзовый даёт с белым фоном контраст
    # 2.74:1 — ниже порога 3:1, и опознавать сектор по одному цвету
    # было бы ненадёжно.
    _plain_labels(chart)
    chart.dataLabels.showPercent = True
    chart.dataLabels.showCatName = True
    chart.dataLabels.showVal = False
    chart.legend = None

    series = chart.series[0]
    series.data_points = []
    for index, (_name, _key, color) in enumerate(MACROS):
        point = DataPoint(idx=index)
        point.graphicalProperties.solidFill = color
        # Белая обводка в два пункта разделяет соседние сектора: без неё
        # граница между ними читается как ещё один цвет.
        point.graphicalProperties.line.solidFill = "FFFFFF"
        point.graphicalProperties.line.width = 25400
        series.data_points.append(point)

    return chart


def _chart_macro_grams(sheet: Worksheet, first_row: int) -> BarChart:
    """Норма и факт по белкам, жирам и углеводам — в граммах.

    Все три величины в одних единицах, поэтому помещаются на одну ось.
    Калорий здесь нет намеренно: 2200 ккал и 91 г на общей шкале
    несопоставимы, а вторая ось — самый частый способ соврать графиком.
    """
    chart = BarChart()
    chart.type = "col"
    chart.title = "Норма и факт, граммы в день"
    chart.height, chart.width = 8.0, 12.0

    data = Reference(sheet, min_col=4, max_col=5,
                     min_row=first_row - 1, max_row=first_row + 2)
    labels = Reference(sheet, min_col=1, min_row=first_row, max_row=first_row + 2)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(labels)

    # Отсчёт от нуля. Excel по умолчанию обрезает шкалу под данные,
    # и расхождение в один грамм превращается в вдвое более высокий столбик.
    chart.y_axis.scaling.min = 0
    chart.y_axis.title = None
    chart.x_axis.title = None
    chart.legend.position = "b"
    chart.gapWidth = 60

    # Норма серая и приглушённая, факт — цветной: сравнивают всегда
    # факт с нормой, а не наоборот.
    chart.series[0].graphicalProperties.solidFill = LINE
    chart.series[1].graphicalProperties.solidFill = PROTEIN
    for series in chart.series:
        series.graphicalProperties.line.noFill = True

    return chart


def _chart_daily_kcal(by_day: Worksheet, days: int, target_kcal: float) -> LineChart:
    """Как калорийность гуляет по дням относительно нормы."""
    chart = LineChart()
    chart.title = "Калорийность по дням"
    chart.height, chart.width = 8.0, 23.0

    data = Reference(by_day, min_col=2, max_col=3, min_row=1, max_row=days + 1)
    labels = Reference(by_day, min_col=1, min_row=2, max_row=days + 1)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(labels)

    # Шкала прибита к норме ±20%. Без этого Excel растягивает разброс
    # в полтора процента на всю высоту графика, и ровный рацион
    # выглядит как американские горки.
    # Границы округляются до сотен, иначе на оси появляются подписи
    # вида 1182 и 1382 — читать такую шкалу неудобно.
    chart.y_axis.scaling.min = int(target_kcal * 0.8 // 100 * 100)
    chart.y_axis.scaling.max = int(-(-target_kcal * 1.2 // 100) * 100)
    chart.y_axis.title = "ккал"
    chart.x_axis.title = "день"
    chart.legend.position = "b"

    fact, norm = chart.series[0], chart.series[1]
    fact.graphicalProperties.line.solidFill = PROTEIN
    fact.graphicalProperties.line.width = 25400
    fact.smooth = False

    # Норма — пунктир: это ориентир, а не измерение.
    norm.graphicalProperties.line.solidFill = MUTED
    norm.graphicalProperties.line.width = 19050
    norm.graphicalProperties.line.dashStyle = "dash"
    norm.smooth = False

    return chart


# ────────────────────────────────────────────────────────────
# Лист «Сводка»
# ────────────────────────────────────────────────────────────


def _sheet_summary(
    book: Workbook, menu: Menu, by_day: Worksheet, profile: dict[str, Any] | None
) -> Worksheet:
    sheet = book.create_sheet("Сводка", 0)
    fact = _averages(menu)
    targets = menu.targets

    sheet["A1"] = f"Меню на {len(menu.days)} дн."
    sheet["A1"].font = TITLE_FONT

    if profile:
        sheet["A2"] = _describe(profile)
        sheet["A2"].font = MUTED_FONT

    # ── Таблица «норма / факт / отклонение» ──────────────────
    _header_row(sheet, 4, ["Показатель", "Норма в день", "Факт в среднем", "Отклонение"])

    rows = [
        ("Калории, ккал", "kcal"),
        ("Белки, г", "protein_g"),
        ("Жиры, г", "fat_g"),
        ("Углеводы, г", "carb_g"),
        ("Клетчатка, г", "fiber_g"),
        ("Натрий, мг (не более)", "sodium_mg"),
    ]

    for offset, (label, key) in enumerate(rows):
        row = 5 + offset
        # У натрия в нормах цели нет — есть верхняя граница. Показывать
        # её как «норму» неправильно, но и оставлять пустую клетку хуже:
        # читатель не поймёт, много 2052 мг или мало.
        target = SODIUM_LIMIT_MG if key == "sodium_mg" else targets.get(key)
        actual = fact.get(key, 0.0)

        sheet.cell(row=row, column=1, value=label).border = BOX
        cell_target = sheet.cell(row=row, column=2,
                                 value=round(target) if target else None)
        cell_actual = sheet.cell(row=row, column=3, value=round(actual))
        cell_ratio = sheet.cell(row=row, column=4,
                                value=(actual / target) if target else None)

        for cell in (cell_target, cell_actual, cell_ratio):
            cell.border = BOX
            cell.number_format = "0"
        cell_ratio.number_format = "0%"

    # ── Данные для диаграмм ──────────────────────────────────
    # График в Excel всегда ссылается на ячейки, а не хранит числа внутри,
    # поэтому исходные числа лежат тут же. Одна колонка категорий кормит
    # обе диаграммы: и круговую, и столбчатую.
    head = 13
    for column, title in enumerate(
        ["Макронутриент", "Калорий в день", "", "Норма, г", "Факт, г"], start=1
    ):
        sheet.cell(row=head, column=column, value=title or None).font = MUTED_FONT

    for offset, (label, key, _color) in enumerate(MACROS):
        row = head + 1 + offset
        sheet.cell(row=row, column=1, value=label)
        sheet.cell(row=row, column=2,
                   value=round(fact[key] * KCAL_PER_GRAM[key])).number_format = "0"
        sheet.cell(row=row, column=4,
                   value=round(targets.get(key, 0))).number_format = "0"
        sheet.cell(row=row, column=5,
                   value=round(fact[key])).number_format = "0"

    sheet.add_chart(_chart_macro_split(sheet, head + 1), "G4")
    sheet.add_chart(_chart_macro_grams(sheet, head + 1), "O4")
    sheet.add_chart(
        _chart_daily_kcal(by_day, len(menu.days), targets["kcal"]), "A21")

    _fit_columns(sheet, {"A": 22, "B": 14, "C": 16, "D": 13, "E": 11})

    return sheet


def _years(age: int) -> str:
    """«21 год», «22 года», «31 год», «45 лет»."""
    if 11 <= age % 100 <= 14:
        return f"{age} лет"
    return f"{age} " + {1: "год", 2: "года", 3: "года", 4: "года"}.get(age % 10, "лет")


def _describe(profile: dict[str, Any]) -> str:
    """Строка о человеке под заголовком."""
    sex = {"male": "мужчина", "female": "женщина"}.get(profile.get("sex", ""), "")
    parts = [part for part in (
        sex,
        _years(profile["age"]) if profile.get("age") else "",
        f"{profile['height_cm']:.0f} см" if profile.get("height_cm") else "",
        f"{profile['weight_kg']:.0f} кг" if profile.get("weight_kg") else "",
        GOAL_LABELS.get(profile.get("goal", ""), ""),
    ) if part]

    line = ", ".join(parts)
    if profile.get("exclude"):
        line += f" · исключено: {describe_restrictions(profile['exclude'])}"
    return line


# ────────────────────────────────────────────────────────────
# Точки входа
# ────────────────────────────────────────────────────────────


def menu_to_workbook(menu: Menu, profile: dict[str, Any] | None = None) -> Workbook:
    """Собрать книгу Excel по меню."""
    book = Workbook()
    # Workbook() всегда создаёт пустой лист, а листы здесь заводятся явно.
    book.remove(book.active)

    by_day = _sheet_by_day(book, menu)
    _sheet_menu(book, menu)
    _sheet_summary(book, menu, by_day, profile)

    book.active = 0
    return book


def menu_to_xlsx(menu: Menu, profile: dict[str, Any] | None = None) -> bytes:
    """Готовые байты файла — то, что отдаётся по HTTP."""
    buffer = io.BytesIO()
    menu_to_workbook(menu, profile).save(buffer)
    return buffer.getvalue()


if __name__ == "__main__":  # pragma: no cover
    from pathlib import Path

    from foods import load_catalog
    from solver import build_menu
    from targets import Profile, compute_targets

    person = Profile(sex="female", age=31, height_cm=168, weight_kg=74,
                     activity="light", goal="lose")
    sample = build_menu(load_catalog(), compute_targets(person), days=7, seed=0)

    out = Path("menu.xlsx")
    out.write_bytes(menu_to_xlsx(sample, person.model_dump()))
    print(f"Сохранено → {out.resolve()}")
