# Агент-нутрициолог: сервис + веб-интерфейс в одном контейнере.
#
# Индекс и каталог кладутся в образ ГОТОВЫМИ, а не собираются внутри:
# сборка индекса требует ключа API и стоит денег, делать это на каждый билд
# бессмысленно. Нужные в рантайме артефакты весят ~45 МБ, всё остальное
# отсекает .dockerignore.
#
# Сборка и запуск:
#     docker compose up --build

FROM python:3.12-slim

# uv ставит зависимости на порядок быстрее pip.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# ── Слой зависимостей ─────────────────────────────────────────
# Ставится до копирования кода: пока pyproject.toml не менялся, Docker
# переиспользует этот слой, и правка агента не тянет переустановку пакетов.
#
# `package = false` в pyproject говорит uv, что проект — не библиотека,
# а набор модулей: ставятся только зависимости, сам код не упаковывается.
# uv.lock копируется вместе с pyproject и фиксирует версии всех 110 пакетов.
# Без него `uv sync` резолвит зависимости заново на каждой сборке: образ,
# собранный сегодня и через месяц, содержал бы разные версии — и «у меня
# работало» переставало быть проверяемым утверждением.
COPY pyproject.toml uv.lock ./

# --frozen запрещает молча обновлять замок: если он разошёлся с pyproject,
# сборка падает, а не подсовывает другие версии.
RUN uv sync --no-dev --no-install-project --frozen

# ── Код и артефакты ───────────────────────────────────────────
COPY . .

# Проверка, что артефакты доехали. Без неё ошибка в .dockerignore привела бы
# к тому, что сервис поднимется и будет молча отвечать «ничего не нашёл».
# Упасть на сборке честнее.
RUN test -d lance_db/vectorstore \
      || (echo "НЕТ ИНДЕКСА: выполните python build_index.py до сборки образа" && exit 1)
RUN test -f data/processed/foods.parquet \
      || (echo "НЕТ КАТАЛОГА: выполните python foods.py --rebuild до сборки образа" && exit 1)
RUN test -f data/processed/ingredients.parquet \
      || (echo "НЕТ СОСТАВА: выполните python ingredients.py --rebuild до сборки образа" && exit 1)
# Без полного состава ограничения проверяются по усечённому — сервис
# поднимется, а свинина во вложенных рецептурах пройдёт молча.
RUN test -f data/processed/ingredient_leaves.parquet \
      || (echo "НЕТ ПОЛНОГО СОСТАВА: выполните python ingredients.py --rebuild-leaves" && exit 1)
# Веса единиц USDA для «съел не по плану»: без них «два куска пиццы»
# считались как две целых пиццы (portions.py).
RUN test -f data/processed/unit_portions.parquet \
      || (echo "НЕТ ВЕСОВ ПОРЦИЙ: выполните python portions.py --rebuild" && exit 1)
# Поиск съеденного по смыслу. Без файла сервис работает — ищет по словам,
# но заметно хуже (food_search.py), поэтому тоже обязателен.
RUN test -f lance_db/catalog_names.npz \
      || (echo "НЕТ ВЕКТОРОВ НАЗВАНИЙ: выполните python food_search.py --rebuild" && exit 1)

# ── Права ─────────────────────────────────────────────────────
# Сервис смотрит наружу портом, поэтому работает не от root. Вместе
# с read_only и no-new-privileges в compose это значит, что даже при
# ошибке в коде процессу нечего перезаписать и некуда повыситься.
#
# Пользователь заводится ПОСЛЕ копирования файлов: иначе COPY положил бы
# их от root, а сервису они нужны только на чтение — так и оставляем.
RUN useradd --create-home --uid 10001 agent

# Сервис пишет только в два тома: журнал чата (chat_log.py) и хранилище
# разговоров (storage.py, SQLite). Папки заводятся в образе с владельцем
# agent: именованный том при первом подключении наследует их права. Без
# этого том принадлежал бы root — журнал молча выключился бы на первой
# записи, а хранилище уронило бы запуск.
RUN mkdir -p /app/logs /app/state && chown agent /app/logs /app/state
USER agent

EXPOSE 8000

# start-period с запасом: на старте открывается индекс LanceDB и строится
# BM25 по всему корпусу. Без запаса оркестратор убьёт контейнер до прогрева.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).status == 200 else 1)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
