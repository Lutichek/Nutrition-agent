"""
Тесты эмбеддера.

Проверяется не векторизация — она требует сети, — а поведение кэша.
Именно оно роняло сервис в контейнере.
"""

from __future__ import annotations

from embedder import Embedder


class TestCacheIsOptional:
    """Кэш эмбеддингов не должен быть условием запуска.

    Наблюдалось: папка кэша исключена из образа .dockerignore, файловая
    система контейнера смонтирована только на чтение — и `mkdir` в
    `Embedder.__init__` падал с «Read-only file system» ещё до первого
    запроса. Сервис не поднимался вовсе, хотя кэш ему не нужен:
    он существует, чтобы не переплачивать при пересборке индекса,
    а внутри контейнера пересборки не бывает.
    """

    def test_unwritable_directory_disables_cache_instead_of_crashing(self, tmp_path):
        blocked = tmp_path / "файл-а-не-папка"
        blocked.write_text("", encoding="utf-8")

        # Создать папку внутри файла нельзя — та же ошибка по сути,
        # что и запись на файловую систему только для чтения.
        embedder = Embedder(client=None, cache_dir=blocked / "кэш")
        assert embedder.cache_dir is None

    def test_cache_can_be_turned_off_explicitly(self):
        assert Embedder(client=None, cache_dir=None).cache_dir is None

    def test_writable_directory_is_used(self, tmp_path):
        embedder = Embedder(client=None, cache_dir=tmp_path / "кэш")
        assert embedder.cache_dir is not None
        assert embedder.cache_dir.is_dir()

    def test_reads_and_writes_survive_a_disabled_cache(self):
        embedder = Embedder(client=None, cache_dir=None)
        assert embedder._cache_get("текст") is None
        embedder._cache_put("текст", [0.1, 0.2])   # не должно бросить
