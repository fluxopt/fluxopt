from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from fluxopt.types import ProfileRef, align, compute_dt, normalize_timesteps

TIME = pl.Series('time', [datetime(2024, 1, 1, h) for h in range(3)], dtype=pl.Datetime('us'))
PERIOD = pl.Series('period', [2024, 2030], dtype=pl.Int64)
FLOW = pl.Series('flow', ['a', 'b'])


class TestNormalizeTimesteps:
    def test_datetime_list(self):
        result = normalize_timesteps([datetime(2024, 1, 1, h) for h in range(3)])
        assert isinstance(result, pl.Series)
        assert result.name == 'time'
        assert len(result) == 3

    def test_polars_series(self):
        series = pl.datetime_range(datetime(2024, 1, 1), datetime(2024, 1, 1, 2), '1h', eager=True)
        assert normalize_timesteps(series).to_list() == TIME.to_list()

    @pytest.mark.parametrize(
        'timesteps',
        [
            pytest.param([0, 1, 2], id='int-list'),
            pytest.param(pl.Series([0, 1, 2]), id='int-series'),
            pytest.param(['t0', 't1', 't2'], id='string-list'),
            pytest.param([1.0, 2.0, 3.0], id='float-list'),
            pytest.param([False, True], id='bool-list'),
            pytest.param([datetime(2024, 1, 1), 1], id='mixed'),
        ],
    )
    def test_anything_but_timestamps_is_refused(self, timesteps):
        with pytest.raises(TypeError, match=r'must be timestamps.*pl\.datetime_range'):
            normalize_timesteps(timesteps)

    def test_empty_list_rejected(self):
        with pytest.raises(ValueError, match='must not be empty'):
            normalize_timesteps([])

    def test_non_monotonic_datetimes_rejected(self):
        with pytest.raises(ValueError, match='monotonically increasing'):
            normalize_timesteps([datetime(2024, 1, 1, 2), datetime(2024, 1, 1, 0)])

    def test_duplicate_datetimes_rejected(self):
        with pytest.raises(ValueError, match='duplicates'):
            normalize_timesteps([datetime(2024, 1, 1), datetime(2024, 1, 1)])


class TestComputeDt:
    def test_explicit_scalar(self):
        assert compute_dt(TIME, 0.5).tolist() == [0.5, 0.5, 0.5]

    def test_explicit_list(self):
        assert compute_dt(TIME, [1.0, 2.0, 3.0]).tolist() == [1.0, 2.0, 3.0]

    def test_explicit_list_of_the_wrong_length(self):
        with pytest.raises(ValueError, match='does not match'):
            compute_dt(TIME[:2], [1.0, 2.0, 3.0])

    def test_derived_from_the_steps(self):
        assert compute_dt(TIME, None).tolist() == [1.0, 1.0, 1.0]

    def test_uneven_steps_take_the_gap_to_the_next(self):
        time = pl.Series('time', [datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 1), datetime(2024, 1, 1, 4)])
        assert compute_dt(time, None).tolist() == [1.0, 1.0, 3.0], 'the first step takes the second one'

    def test_single_step_lasts_an_hour(self):
        assert compute_dt(TIME[:1], None).tolist() == [1.0]


class TestAlignScalar:
    def test_fills_every_axis(self):
        result = align(2.0, {'flow': FLOW, 'time': TIME})
        assert result.shape == (2, 3)
        assert np.all(result == 2.0)

    def test_int(self):
        assert align(3, {'time': TIME}).tolist() == [3.0, 3.0, 3.0]


class TestAlign1d:
    @pytest.mark.parametrize(
        'value',
        [
            pytest.param([1.0, 2.0, 3.0], id='list'),
            pytest.param(np.array([1.0, 2.0, 3.0]), id='ndarray'),
            pytest.param(pl.Series([1.0, 2.0, 3.0]), id='series'),
        ],
    )
    def test_matched_by_length(self, value):
        assert align(value, {'time': TIME}).tolist() == [1.0, 2.0, 3.0]

    def test_broadcast_over_the_other_axes(self):
        result = align([10.0, 20.0, 30.0], {'flow': FLOW, 'time': TIME})
        assert result.tolist() == [[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]], 'axis order is the order of axes'

    def test_time_wins_a_tie(self):
        result = align([1.0, 2.0], {'time': TIME[:2], 'period': PERIOD})
        assert result.tolist() == [[1.0, 1.0], [2.0, 2.0]]

    def test_ambiguous_length_raises(self):
        with pytest.raises(ValueError, match='several dimensions'):
            align([1.0, 2.0], {'flow': FLOW, 'period': PERIOD})

    def test_no_match_raises(self):
        with pytest.raises(ValueError, match='does not match any dimension'):
            align([1.0, 2.0], {'time': TIME})

    def test_two_dimensional_array_raises(self):
        with pytest.raises(ValueError, match='must be 1-D'):
            align(np.ones((3, 2)), {'time': TIME, 'period': PERIOD})


class TestAlignTable:
    def test_tidy_table_on_two_axes(self):
        table = pl.DataFrame(
            {
                'period': [2030, 2024, 2030, 2024, 2030, 2024],
                'time': [t for t in TIME.to_list() for _ in range(2)],
                'value': [1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
            }
        )
        result = align(table, {'time': TIME, 'period': PERIOD})
        assert result.tolist() == [[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]], 'rows land by label, not by position'

    def test_broadcast_over_the_axes_it_has_no_column_for(self):
        table = pl.DataFrame({'period': [2024, 2030], 'value': [1.0, 2.0]})
        assert align(table, {'time': TIME, 'period': PERIOD}).tolist() == [[1.0, 2.0]] * 3

    @pytest.mark.parametrize(
        ('table', 'message'),
        [
            pytest.param(pl.DataFrame({'period': [2024, 2030]}), "needs a 'value' column", id='no-value'),
            pytest.param(pl.DataFrame({'scenario': ['a'], 'value': [1.0]}), 'not dimensions here', id='foreign-column'),
            pytest.param(
                pl.DataFrame({'period': [2024, 2031], 'value': [1.0, 2.0]}),
                'labels the model does not',
                id='unknown-label',
            ),
            pytest.param(
                pl.DataFrame({'period': [2024, 2024], 'value': [1.0, 2.0]}), 'repeats a combination', id='duplicate'
            ),
            pytest.param(pl.DataFrame({'period': [2024], 'value': [1.0]}), 'give a value for each', id='missing'),
        ],
    )
    def test_a_table_that_does_not_fill_its_axes_is_refused(self, table, message):
        with pytest.raises(ValueError, match=message):
            align(table, {'time': TIME, 'period': PERIOD})


class TestAlignRefused:
    def test_unresolved_profile_ref(self):
        with pytest.raises(ValueError, match='Unresolved ProfileRef'):
            align(ProfileRef(table='p', column='x'), {'time': TIME})

    @pytest.mark.parametrize('value', [pytest.param({}, id='dict'), pytest.param(True, id='bool')])
    def test_not_a_variate(self, value):
        with pytest.raises(TypeError, match='Unsupported Variate type'):
            align(value, {'time': TIME})


class TestProfileRef:
    def test_a_table_with_keys_resolves_to_a_tidy_table(self):
        table = pl.DataFrame({'time': TIME, 'gas': [1.0, 2.0, 3.0], 'power': [0.0, 0.0, 0.0]})
        resolved = ProfileRef(table='prices', column='gas').resolve({'prices': table})
        assert isinstance(resolved, pl.DataFrame)
        assert resolved.columns == ['time', 'value']
        assert align(resolved, {'time': TIME}).tolist() == [1.0, 2.0, 3.0]
