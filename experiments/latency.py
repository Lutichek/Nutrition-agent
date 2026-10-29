"""
Задержка и токены ответа агента по шагам — на живой модели.

Зачем. Журнал чата показал, что ответ с планом идёт 12-31 с, и почти всё
время — последовательные вызовы модели и сборка меню. Правки скорости
меняют ПОРЯДОК работы, а не числа, поэтому им нужен свой замер: метрики
качества RAG тут ничего не скажут, а время по одной реплике дрожит в разы
(«понять запрос» в журнале — от 1.1 до 7.9 с на одном и том же промпте).

Что меряется. Фиксированный набор реплик: план на 1, 7 и 30 дней и два
вопроса по статьям, каждый в новом разговоре, по ``--repeats`` прогонов.
По каждому шагу — секунды; по каждому вызову модели — токены входа
и выхода, если провайдер их вернул (нет счётчика — пропуск, не ноль).

Сравнение — только попеременными прогонами в одно время: задержка
провайдера плывёт за час сильнее, чем меняет любая из правок.

Запуск::

    python -m experiments.latency --save до
    python -m experiments.latency --against до

Стоит денег: ~60 вызовов модели на прогон (gpt-4o-mini — копейки).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import ConversationState, build_agent  # noqa: E402
from providers import get_clients  # noqa: E402

RESULTS_DIR = ROOT / "_research" / "latency"

PROFILE = "Мне 35, мужчина, рост 180 см, вес 85 кг, тренируюсь 3 раза в неделю, хочу похудеть."

CASES = {
    "план 1 день": f"{PROFILE} Составь рацион.",
    "план 7 дней": f"{PROFILE} Составь рацион на неделю.",
    "план 30 дней": f"{PROFILE} Составь меню на месяц.",
    "вопрос: белок": "Сколько белка нужно при похудении?",
    "вопрос: клетчатка": "Помогает ли клетчатка снизить холестерин?",
}


def run_case(agent, message: str, stream: bool = True) -> dict:
    # Первый текст — то, ради чего ответ идёт по частям: сколько человек
    # смотрит на «думаю…». Без потока он совпадает с полным временем.
    first_text: list[float] = []

    def on_event(event: dict) -> None:
        if event["type"] == "text" and not first_text:
            first_text.append(time.perf_counter() - started)

    started = time.perf_counter()
    answer = agent.run(message, state=ConversationState(),
                       on_event=on_event if stream else None)
    total = time.perf_counter() - started

    nodes: dict[str, float] = {}
    prompt_tokens: list[int] = []
    completion_tokens: list[int] = []
    for step in answer.trace.get("nodes", []):
        nodes[step["node"]] = nodes.get(step["node"], 0.0) + step["seconds"]
        for call in step.get("llm") or []:
            if "prompt_tokens" in call:
                prompt_tokens.append(call["prompt_tokens"])
            if "completion_tokens" in call:
                completion_tokens.append(call["completion_tokens"])
    return {
        "seconds": round(total, 2),
        "first_text": round(first_text[0], 2) if first_text else round(total, 2),
        "nodes": {name: round(sec, 2) for name, sec in nodes.items()},
        # None, если провайдер не вернул ни одного счётчика: «0» читалось бы
        # как бесплатный ответ.
        "prompt_tokens": sum(prompt_tokens) if prompt_tokens else None,
        "completion_tokens": sum(completion_tokens) if completion_tokens else None,
        "answer_chars": len(answer.answer),
        "intent": answer.trace.get("intent"),
    }


def summarise(runs: dict[str, list[dict]]) -> dict[str, dict]:
    out = {}
    for case, items in runs.items():
        def med(key, items=items):
            values = [item[key] for item in items if item[key] is not None]
            return statistics.median(values) if values else None

        node_names = sorted({name for item in items for name in item["nodes"]})
        out[case] = {
            "seconds": med("seconds"),
            "first_text": med("first_text"),
            "prompt_tokens": med("prompt_tokens"),
            "completion_tokens": med("completion_tokens"),
            "answer_chars": med("answer_chars"),
            "nodes": {
                name: statistics.median(item["nodes"].get(name, 0.0) for item in items)
                for name in node_names
            },
            "intents": sorted({str(item["intent"]) for item in items}),
        }
    return out


def fmt(value, unit=""):
    if value is None:
        return "—"
    return f"{value:.1f}{unit}" if isinstance(value, float) else f"{value}{unit}"


def print_report(summary: dict, baseline: dict | None) -> None:
    for case, row in summary.items():
        base = (baseline or {}).get(case)
        print(f"\n{case}  (маршрут: {', '.join(row['intents'])})")
        for key, label, unit in (("seconds", "всего", " с"),
                                 ("first_text", "первый текст", " с"),
                                 ("prompt_tokens", "токенов вход", ""),
                                 ("completion_tokens", "токенов выход", ""),
                                 ("answer_chars", "символов ответа", "")):
            line = f"  {label:<18} {fmt(row[key], unit):>10}"
            if base and base.get(key) is not None and row[key] is not None:
                line += f"   было {fmt(base[key], unit):>10}   {row[key] - base[key]:+.1f}"
            print(line)
        for name, sec in row["nodes"].items():
            line = f"    {name:<16} {sec:6.2f} с"
            if base and name in base["nodes"]:
                line += f"   было {base['nodes'][name]:6.2f}"
            print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--save", help="сохранить сводку под этим именем")
    parser.add_argument("--against", help="сравнить с сохранённой сводкой")
    parser.add_argument("--cases", nargs="*", help="только эти случаи (по началу названия)")
    parser.add_argument("--no-speculate", action="store_true",
                        help="без вызовов наперёд — для сравнения с ними")
    parser.add_argument("--no-stream", action="store_true",
                        help="без ответа по частям — для сравнения с ним")
    parser.add_argument("--free-json-extract", action="store_true",
                        help="извлечение профиля свободным JSON, а не по схеме")
    args = parser.parse_args()

    llm_client, model, embed_client = get_clients("polza")
    # Как в сервисе (api.py): меряется то, что ждёт человек.
    agent = build_agent(llm_client, model=model, embed_client=embed_client,
                        speculate=not args.no_speculate,
                        structured_extract=not args.free_json_extract)

    cases = {name: text for name, text in CASES.items()
             if not args.cases or any(name.startswith(c) for c in args.cases)}
    runs: dict[str, list[dict]] = {name: [] for name in cases}
    # Случаи чередуются внутри повтора, а не идут блоками: так дрейф
    # провайдера за время прогона ложится на все случаи поровну.
    for repeat in range(args.repeats):
        for name, text in cases.items():
            result = run_case(agent, text, stream=not args.no_stream)
            runs[name].append(result)
            print(f"[{repeat + 1}/{args.repeats}] {name}: {result['seconds']:.1f} с", flush=True)

    summary = summarise(runs)
    baseline = None
    if args.against:
        baseline = json.loads((RESULTS_DIR / f"{args.against}.json").read_text(encoding="utf-8"))
    print_report(summary, baseline)

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"{args.save}.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (RESULTS_DIR / f"{args.save}_runs.json").write_text(
            json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nСохранено: {path}")


if __name__ == "__main__":
    main()
