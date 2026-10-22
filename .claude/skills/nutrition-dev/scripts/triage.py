"""Разбор журнала чата: какие реплики стоит посмотреть и почему.

Запуск из корня проекта:

    python <skill>/scripts/triage.py logs/chat-2026-10-20.jsonl
    python <skill>/scripts/triage.py logs/chat-2026-10-20.jsonl --turn 3   # одна запись подробно

Модель не вызывается: план повторяется через узел make_plan
(chat_log.replay_plan), ограничения проверяются текущим фильтром.
Расхождение повтора с записью значит «с тех пор поменялся солвер или
каталог», а не «баг не воспроизводится».
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from agent import NutritionAgent  # noqa: E402
from chat_log import replay_plan  # noqa: E402
from foods import load_catalog  # noqa: E402
from ingredients import LEAVES_PATH  # noqa: E402
from restrictions import matches_terms  # noqa: E402
from solver import _filter_catalog  # noqa: E402

QUESTION = re.compile(r"\?|^(?:сколько|почему|зачем|можно ли|полезн|вредн|что)", re.I)
SLOW_MS = 20_000


def flags_for(record: dict, catalog, names, leaves, agent) -> list[str]:
    flags: list[str] = []
    trace = record.get("trace") or {}
    if record.get("status") == "error":
        flags.append(f"ОШИБКА: {record.get('error', '')[:120]}")
    if record.get("elapsed_ms", 0) > SLOW_MS:
        flags.append(f"медленно: {record['elapsed_ms'] / 1000:.0f} с")

    shown = record.get("shown") or record.get("answer") or ""
    if "PMID" in shown.upper() or re.search(r"например\s+и\b", shown):
        flags.append("в показанном тексте остатки ссылок")

    message = record.get("message", "")
    if trace.get("intent") == "plan" and QUESTION.search(message) and not re.search(r"\d", message):
        flags.append("вопрос ушёл в ветку плана — проверь маршрут")

    plan_days = trace.get("plan_days")
    profile = trace.get("profile") or {}
    if plan_days:
        exclude = profile.get("exclude") or []
        allowed = set(_filter_catalog(catalog, exclude)["fdc_id"])
        for day, ids in enumerate(plan_days, start=1):
            for fdc_id in ids:
                if fdc_id in allowed:
                    continue
                why = ""
                if exclude:
                    # Как у фильтра каталога — с исключениями (matches_terms).
                    own = leaves.loc[leaves["fdc_id"] == fdc_id, "name_en"].astype(object)
                    hit = own[matches_terms(own, exclude, skip_absent=True).to_numpy()]
                    why = f" — в составе: {', '.join(hit[:2])}" if len(hit) else ""
                flags.append(f"день {day}: «{names.get(fdc_id, fdc_id)}» сейчас не прошло бы "
                             f"фильтр (exclude={exclude}){why}")
        try:
            again = replay_plan(agent, trace)
            if again != plan_days:
                flags.append("повтор плана даёт другое меню: солвер или каталог изменились "
                             "после этой записи")
        except Exception as error:  # noqa: BLE001
            flags.append(f"повтор не удался: {type(error).__name__}: {error}")
    return flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="+")
    parser.add_argument("--turn", type=int, help="подробно одну запись (номер строки, с 1)")
    args = parser.parse_args()

    catalog = load_catalog()
    names = dict(zip(catalog["fdc_id"], catalog["name"], strict=True))
    leaves = pd.read_parquet(LEAVES_PATH)
    agent = NutritionAgent(client=None, model="replay", retriever=None, catalog=catalog)

    records = [json.loads(line) for path in args.log
               for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for number, record in enumerate(records, start=1):
        if args.turn and number != args.turn:
            continue
        trace = record.get("trace") or {}
        head = (f"#{number} {record.get('ts', '')} сессия {str(record.get('session_id'))[:8]} "
                f"реплика {record.get('turn')} [{trace.get('intent')}] "
                f"«{record.get('message', '')[:70]}»")
        flags = flags_for(record, catalog, names, leaves, agent)
        print(head)
        for flag in flags or ["— замечаний нет"]:
            print("    •", flag)
        if args.turn:
            print(json.dumps(trace, ensure_ascii=False, indent=1)[:4000])


if __name__ == "__main__":
    main()
