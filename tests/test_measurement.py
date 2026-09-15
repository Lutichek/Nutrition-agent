"""
Тесты парного бутстрэпа — того самого, по которому принимаются решения.

Наблюдавшийся случай. Сравнение двух промптов генерации напечатало:

    faithfulness  +nan  [+nan, +nan]  в пределах шума

и следом автоматический вердикт объявил «faithfulness ПРОСЕЛА». Ни того,
ни другого в данных не было: faithfulness просто не посчиталась у пяти
вопросов из 210 (там не нашлось контекста, проверять было нечего).
Один NaN в разнице отравлял всё среднее, а вердикт сравнивал NaN с нулём,
получал `False` и печатал приговор.

Это тот же класс дефекта, что и `fillna(0.0)` в evaluation.py: **отсутствие
оценки превращается в оценку**. Только здесь оно превращалось сразу
в решение о продукте.

Тогда обошлось — правку отклонили по correctness, которая посчиталась
честно. Но пройди correctness порог, NaN заблокировал бы верную правку.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from measurement import paired_bootstrap


class TestMissingValuesDoNotPoisonTheComparison:
    def test_one_nan_does_not_make_the_whole_interval_nan(self):
        """Ровно наблюдавшийся случай: одна дыра из шести вопросов."""
        base = pd.Series([1.0, 0.5, 1.0, np.nan, 1.0, 0.5])
        variant = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 0.5])

        mean, low, high = paired_bootstrap(base, variant)

        assert not np.isnan(mean), "один пропуск обнулил весь замер"
        assert not np.isnan(low) and not np.isnan(high)

    def test_pair_is_dropped_from_both_sides(self):
        """Выпадает пара целиком, а не одна её половина.

        Иначе сравнение поедет: у базовой конфигурации останется шесть
        вопросов, у варианта пять, и разница будет считаться между
        разными наборами.
        """
        base = pd.Series([1.0, 0.0, np.nan, 1.0])
        variant = pd.Series([1.0, 0.0, 1.0, 1.0])
        mean, _low, _high = paired_bootstrap(base, variant)
        assert mean == pytest.approx(0.0)

    def test_missing_on_the_variant_side_too(self):
        base = pd.Series([1.0, 1.0, 1.0, 1.0])
        variant = pd.Series([1.0, np.nan, 1.0, 1.0])
        mean, _low, _high = paired_bootstrap(base, variant)
        assert mean == pytest.approx(0.0)

    def test_nothing_measured_at_all_is_nan(self):
        """Если не осталось ни одной пары — честный NaN, а не ноль.

        Ноль означал бы «сравнили, разницы нет»; здесь сравнивать нечего.
        """
        base = pd.Series([np.nan, np.nan])
        variant = pd.Series([1.0, 1.0])
        mean, low, high = paired_bootstrap(base, variant)
        assert np.isnan(mean) and np.isnan(low) and np.isnan(high)


class TestBootstrapItself:
    def test_identical_series_give_zero_difference(self):
        values = pd.Series([1.0, 0.5, 0.0, 1.0, 0.5])
        mean, low, high = paired_bootstrap(values, values)
        assert mean == 0.0 and low == 0.0 and high == 0.0

    def test_constant_improvement_is_detected(self):
        base = pd.Series([0.0] * 40)
        variant = pd.Series([1.0] * 40)
        mean, low, high = paired_bootstrap(base, variant)
        assert mean == pytest.approx(1.0)
        assert low > 0, "постоянное улучшение обязано быть значимым"

    def test_result_is_reproducible(self):
        rng = np.random.default_rng(0)
        base = pd.Series(rng.random(50))
        variant = pd.Series(rng.random(50))
        assert paired_bootstrap(base, variant) == paired_bootstrap(base, variant)
