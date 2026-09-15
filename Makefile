# Короткие команды разработки. Всё то же самое можно набрать руками —
# см. README, — но помнить точные флаги не обязательно.
#
# На Windows нужен make из Git for Windows или WSL; без него работают
# те же команды напрямую.

.DEFAULT_GOAL := help
.PHONY: help setup check run test lint fix metrics solver docker docker-logs clean         retrieval judge-noise qrels-bias chunking chunking-e2e \n        answer-prompt dataset-scope

PY := python

help:  ## Показать этот список
	@grep -E '^[a-z0-9-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Установить зависимости и инструменты разработки
	uv sync --extra dev

check:  ## Проверить готовность к запуску (без обращений к API)
	$(PY) check_setup.py --offline

run:  ## Поднять сервис на http://127.0.0.1:8000
	uvicorn api:app --host 127.0.0.1 --port 8000

test:  ## Прогнать тесты (сеть не нужна)
	$(PY) -m pytest -q

lint:  ## Линтер
	ruff check .

fix:  ## Линтер с автоисправлением
	ruff check . --fix

metrics:  ## Метрики качества RAG по сохранённым ответам
	$(PY) run_eval.py --metrics

solver:  ## Замер солвера: профили × сиды
	$(PY) -m experiments.measure_solver

retrieval:  ## Сравнить стратегии поиска (dense / гибрид / реранкинг)
	$(PY) -m experiments.retrieval

qrels-bias:  ## Проверить разметку независимым судьёй (реранкер ей родня)
	$(PY) -m experiments.qrels_bias

judge-noise:  ## Собственный шум судьи: порог различимой разницы
	$(PY) -m experiments.judge_noise

chunking:  ## Перебор размера чанка по качеству поиска
	$(PY) -m experiments.chunking

chunking-e2e:  ## Доходит ли выигрыш нарезки до ответа (решающий замер)
	$(PY) -m experiments.chunking_end_to_end

answer-prompt:  ## Промпт генерации: боевой против кандидата
	$(PY) -m experiments.answer_prompt

dataset-scope:  ## Сколько в датасете вопросов вне области продукта
	$(PY) -m experiments.dataset_scope

docker:  ## Собрать образ и поднять контейнер
	docker compose up -d --build

docker-logs:  ## Логи контейнера
	docker compose logs -f

clean:  ## Убрать кэши Python и инструментов
	rm -rf __pycache__ tests/__pycache__ .pytest_cache .ruff_cache
