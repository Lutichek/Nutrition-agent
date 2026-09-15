"""
Замеры, на которых стоят решения проекта. Это доказательства, а не продукт.

Здесь лежат скрипты, каждый из которых отвечает ровно на один вопрос
и печатает ответ в виде таблицы с доверительными интервалами. Результаты
и выводы записаны в ``_research/EVOLUTION.md``; сюда смотрят, когда нужно
перепроверить или повторить замер на новых данных.

В рантайм ничего отсюда не импортируется — пакет исключён из образа
(см. ``.dockerignore``). Общая обвязка (пути, бутстрэп, колонки метрик)
живёт в ``measurement.py`` в корне: её используют и замеры, и ``run_eval``.

Запуск — модулем, из корня проекта::

    python -m experiments.retrieval        # стратегии поиска
    python -m experiments.chunking         # размер чанка: метрики поиска
    python -m experiments.chunking_end_to_end  # ...и доходит ли он до ответа
    python -m experiments.qrels_bias       # не меряет ли разметка сама себя
    python -m experiments.judge_noise      # порог различимости судьи
    python -m experiments.judge_bias       # завышает ли судья свои ответы
    python -m experiments.grade            # строгость отбора фрагментов
    python -m experiments.answer_prompt    # промпт генерации
    python -m experiments.dataset_scope    # засорённость датасета
    python -m experiments.measure_solver   # качество подбора рациона

Именно модулем, а не по пути к файлу: скриптам нужен корень проекта
на ``sys.path``, чтобы видеть ``measurement``, ``agent`` и остальное.
"""
