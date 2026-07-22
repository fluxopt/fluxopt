"""Mathematical correctness tests for piecewise-linear conversion.

Wraps :func:`linopy.piecewise.add_piecewise_formulation`. The new API
auto-selects between LP (convex/concave 2-flow inequality), incremental
(monotonic), and SOS2 formulations.
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fluxopt import Carrier, Converter, Effect, Flow, PiecewiseConversion, Port, Status

from .conftest import ts


class TestPiecewiseConversionValidation:
    def test_dict_form(self):
        c = PiecewiseConversion(points={'fuel': [0, 50, 100], 'Heat': [0, 45, 70]})
        normalized = c._iter_normalized()
        assert [t[0] for t in normalized] == ['fuel', 'Heat']
        assert all(t[2] == '==' for t in normalized)

    def test_tuple_form_with_bound(self):
        c = PiecewiseConversion(points=[('fuel', [0, 50, 100]), ('Heat', [0, 45, 70], '>=')])
        normalized = c._iter_normalized()
        assert normalized[1][2] == '>='

    def test_needs_two_flows(self):
        with pytest.raises(ValueError, match='>=2 flows'):
            PiecewiseConversion(points={'fuel': [0, 1, 2]})

    def test_equal_lengths(self):
        with pytest.raises(ValueError, match='same length'):
            PiecewiseConversion(points={'A': [0, 1, 2], 'B': [0, 1]})

    def test_needs_two_breakpoints(self):
        with pytest.raises(ValueError, match='>=2 breakpoints'):
            PiecewiseConversion(points={'A': [0], 'B': [0]})

    def test_at_most_one_bound(self):
        with pytest.raises(ValueError, match='At most one bounded flow'):
            PiecewiseConversion(
                points=[('A', [0, 1], '<='), ('B', [0, 1], '>=')],
            )

    def test_inequality_requires_two_flows(self):
        with pytest.raises(ValueError, match='Inequality bounds require exactly 2 flows'):
            PiecewiseConversion(
                points=[('A', [0, 1], '>='), ('B', [0, 1]), ('C', [0, 1])],
            )

    def test_lp_requires_bound(self):
        with pytest.raises(ValueError, match="method='lp' requires"):
            PiecewiseConversion(points={'A': [0, 1], 'B': [0, 1]}, method='lp')

    def test_no_duplicate_flows(self):
        with pytest.raises(ValueError, match='duplicate flow'):
            PiecewiseConversion(points=[('A', [0, 1]), ('A', [0, 2])])


class TestConverterPiecewiseValidation:
    def test_mutually_exclusive_with_factors(self):
        with pytest.raises(ValueError, match='mutually exclusive'):
            Converter(
                id='X',
                inputs=[Flow(carrier='A', short_id='a')],
                outputs=[Flow(carrier='B')],
                conversion_factors=[{'a': 1, 'B': -1}],
                conversion=PiecewiseConversion(points={'a': [0, 1], 'B': [0, 1]}),
            )

    def test_unknown_flow_in_curve(self):
        with pytest.raises(ValueError, match='unknown flow'):
            Converter(
                id='X',
                inputs=[Flow(carrier='A', short_id='a')],
                outputs=[Flow(carrier='B')],
                conversion=PiecewiseConversion(points={'a': [0, 1], 'C': [0, 1]}),
            )

    def test_flow_status_forbidden_with_curve_status(self):
        with pytest.raises(ValueError, match='cannot have flow-level status'):
            Converter(
                id='X',
                inputs=[Flow(carrier='A', short_id='a')],
                outputs=[Flow(carrier='B', size=10, relative_rate_min=0.1, status=Status())],
                conversion=PiecewiseConversion(points={'a': [0, 1], 'B': [0, 1]}, status=Status()),
            )


class TestPiecewise:
    def test_two_flow_interpolation(self, optimize):
        """A 2-flow PiecewiseConversion interpolates the output linearly between breakpoints.

        Boiler has efficiency 90% in [0,50] (slope 0.9) and 50% in [50,100]
        (slope 0.5). Demand=5 hits the cheap segment: fuel = 5 / 0.9.
        """
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([0, 5, 0]))]),
                Port(id='GasSrc', imports=[Flow(carrier='Gas', effects_per_flow_hour={'cost': 1})]),
            ],
            converters=[
                Converter(
                    id='Boiler',
                    inputs=[Flow(carrier='Gas', short_id='fuel')],
                    outputs=[Flow(carrier='Heat', size=100)],
                    conversion=PiecewiseConversion(points={'fuel': [0, 50, 100], 'Heat': [0, 45, 70]}),
                ),
            ],
        )
        # heat=5 at t=1 → fuel = 5/0.9 ≈ 5.555 → cost = 5.555 * 1
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 5.0 / 0.9, rtol=1e-5)
        assert_allclose(result.solution['flow--rate'].sel(flow='Boiler(fuel)').values[1], 5.0 / 0.9, atol=1e-5)

    def test_segment_selection_picks_efficient_region(self, optimize):
        """Solver picks the more-efficient segment when demand fits.

        Curve: [0, 30, 100] gas → [0, 30, 70] heat. Slope_lo=1.0, slope_hi≈0.571.
        Demand=20 → fits in low segment with slope=1 → fuel=20.
        Demand=50 → forces high segment → fuel = 30 + (50-30)*70/40 = 65.
        """

        def _run(demand_value: float):
            return optimize(
                timesteps=ts(2),
                carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
                effects=[Effect(id='cost')],
                objective='cost',
                ports=[
                    Port(
                        id='Demand',
                        exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([demand_value, 0]))],
                    ),
                    Port(id='GasSrc', imports=[Flow(carrier='Gas', effects_per_flow_hour={'cost': 1})]),
                ],
                converters=[
                    Converter(
                        id='Boiler',
                        inputs=[Flow(carrier='Gas', short_id='fuel')],
                        outputs=[Flow(carrier='Heat', size=100)],
                        conversion=PiecewiseConversion(points={'fuel': [0, 30, 100], 'Heat': [0, 30, 70]}),
                    ),
                ],
            )

        result_low = _run(20.0)
        assert_allclose(result_low.effect_totals.sel(effect='cost').item(), 20.0, rtol=1e-5)

        result_high = _run(50.0)
        assert_allclose(result_high.effect_totals.sel(effect='cost').item(), 65.0, rtol=1e-5)

    def test_three_flow_chp_joint(self, optimize):
        """3-flow CHP curve with shared interpolation weights.

        At any segment, gas, power, and heat all lie on the same piece. So if
        we constrain power=10 (mid-piece), heat is fixed at 15, and gas at 30.
        """
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier(id='Gas'), Carrier(id='Power'), Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='PowerDmd', exports=[Flow(carrier='Power', size=1, fixed_relative_profile=np.array([10, 0]))]),
                Port(id='HeatDmd', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([15, 0]))]),
                Port(id='GasSrc', imports=[Flow(carrier='Gas', effects_per_flow_hour={'cost': 1})]),
            ],
            converters=[
                Converter(
                    id='CHP',
                    inputs=[Flow(carrier='Gas', short_id='fuel')],
                    outputs=[Flow(carrier='Power', size=100), Flow(carrier='Heat', size=100)],
                    conversion=PiecewiseConversion(
                        points={
                            'fuel': [0, 30, 60, 100],
                            'Power': [0, 10, 22, 40],
                            'Heat': [0, 15, 30, 45],
                        }
                    ),
                ),
            ],
        )
        # At t=0: Power=10 → on segment [(0,0,0)-(30,10,15)] → fuel=30, heat=15
        assert_allclose(result.solution['flow--rate'].sel(flow='CHP(fuel)').values[0], 30.0, atol=1e-5)
        assert_allclose(result.solution['flow--rate'].sel(flow='CHP(Heat)').values[0], 15.0, atol=1e-5)

    def test_time_varying_breakpoints(self, optimize):
        """Breakpoints can vary per timestep (e.g. ambient-dependent COP)."""
        # Slope alternates: t=0 has 0.5 efficiency, t=1 has 1.0 efficiency.
        slope_t = np.array([0.5, 1.0])
        bp_max_fuel = np.array([100.0, 50.0])
        bp_max_heat = bp_max_fuel * slope_t  # [50, 50]

        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([20, 20]))]),
                Port(id='GasSrc', imports=[Flow(carrier='Gas', effects_per_flow_hour={'cost': 1})]),
            ],
            converters=[
                Converter(
                    id='Boiler',
                    inputs=[Flow(carrier='Gas', short_id='fuel')],
                    outputs=[Flow(carrier='Heat', size=100)],
                    conversion=PiecewiseConversion(
                        points={
                            'fuel': [np.array([0.0, 0.0]), bp_max_fuel],
                            'Heat': [np.array([0.0, 0.0]), bp_max_heat],
                        }
                    ),
                ),
            ],
        )
        # At t=0: heat=20 needs fuel=40 (slope 0.5). At t=1: heat=20 needs fuel=20 (slope 1.0).
        # Total cost = 40 + 20 = 60.
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 60.0, rtol=1e-5)


class TestPiecewiseStatus:
    def test_status_gates_curve(self, optimize):
        """PiecewiseConversion.status forces all curve flows to 0 when on=0."""
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([0, 5, 0]))]),
                Port(id='GasSrc', imports=[Flow(carrier='Gas', effects_per_flow_hour={'cost': 1})]),
            ],
            converters=[
                Converter(
                    id='Boiler',
                    inputs=[Flow(carrier='Gas', short_id='fuel')],
                    outputs=[Flow(carrier='Heat', size=100)],
                    conversion=PiecewiseConversion(
                        points={'fuel': [0, 50, 100], 'Heat': [0, 45, 70]},
                        status=Status(effects_per_startup={'cost': 1000}),
                    ),
                ),
            ],
        )
        # Startup cost is high — solver may keep on=1 throughout (free at boundaries).
        # Either way, Heat must be exactly 5 at t=1 and 0 at t=0,2.
        heat = result.solution['flow--rate'].sel(flow='Boiler(Heat)').values
        fuel = result.solution['flow--rate'].sel(flow='Boiler(fuel)').values
        on = result.solution['component--on'].sel(component='Boiler').values
        assert_allclose(heat, [0, 5, 0], atol=1e-5)
        # Status gating: when on=0, every curve flow is pinned to bp_0 (zero here).
        for t in range(3):
            if on[t] < 0.5:
                assert fuel[t] < 1e-5, f't={t}: fuel={fuel[t]} but on={on[t]}'
                assert heat[t] < 1e-5, f't={t}: heat={heat[t]} but on={on[t]}'
        # Demand at t=1 forces on=1.
        assert on[1] > 0.5

    def test_status_running_cost(self, optimize):
        """effects_per_running_hour accrues per timestep when on=1."""
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([0, 5, 0]))]),
                Port(id='GasSrc', imports=[Flow(carrier='Gas', effects_per_flow_hour={'cost': 1})]),
            ],
            converters=[
                Converter(
                    id='Boiler',
                    inputs=[Flow(carrier='Gas', short_id='fuel')],
                    outputs=[Flow(carrier='Heat', size=100)],
                    conversion=PiecewiseConversion(
                        points={'fuel': [0, 50, 100], 'Heat': [0, 45, 70]},
                        status=Status(effects_per_running_hour={'cost': 100}),
                    ),
                ),
            ],
        )
        # Running cost is high — solver must keep on=0 except at t=1 (forced by demand).
        # fuel at t=1 = 5/0.9. Cost = fuel*1 + 100 * (one running hour).
        on = result.solution['component--on'].sel(component='Boiler').values
        assert_allclose(on, [0, 1, 0], atol=1e-5)
        expected_cost = 5.0 / 0.9 + 100.0
        assert_allclose(result.effect_totals.sel(effect='cost').item(), expected_cost, rtol=1e-5)


class TestRedundantStatusWarning:
    """Warn when PiecewiseConversion has Status alongside an all-flows-zero breakpoint."""

    def _build_with_curve(self, curve: PiecewiseConversion):
        """Build ModelData with a single piecewise converter using `curve`."""
        from fluxopt.model_data import ModelData

        return ModelData.build(
            timesteps=ts(3),
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([0, 5, 0]))]),
                Port(id='GasSrc', imports=[Flow(carrier='Gas', effects_per_flow_hour={'cost': 1})]),
            ],
            converters=[
                Converter(
                    id='Boiler',
                    inputs=[Flow(carrier='Gas', short_id='fuel')],
                    outputs=[Flow(carrier='Heat', size=100)],
                    conversion=curve,
                )
            ],
        )

    def test_warns_when_zero_breakpoint_with_status(self):
        """Curve with (0, 0) first breakpoint AND Status -> warn."""
        curve = PiecewiseConversion(
            points={'fuel': [0, 50, 100], 'Heat': [0, 45, 70]},
            status=Status(effects_per_startup={'cost': 1}),
        )
        with pytest.warns(UserWarning, match=r'Boiler.*\(0, \.\.\., 0\) breakpoint'):
            self._build_with_curve(curve)

    def test_warns_when_zero_breakpoint_not_first(self):
        """All-zero point anywhere in the curve (not just first) -> warn (SOS2 allows non-monotonic)."""
        curve = PiecewiseConversion(
            points={'fuel': [50, 0, 100], 'Heat': [45, 0, 70]},
            method='sos2',
            status=Status(effects_per_startup={'cost': 1}),
        )
        with pytest.warns(UserWarning, match=r'Boiler.*\(0, \.\.\., 0\) breakpoint'):
            self._build_with_curve(curve)

    def test_no_warn_when_curve_avoids_zero(self):
        """Curve that never hits all-flows-zero -> no warning, even with Status."""
        curve = PiecewiseConversion(
            points={'fuel': [30, 70, 100], 'Heat': [22.5, 58.5, 78.5]},
            status=Status(effects_per_startup={'cost': 1}),
        )
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            self._build_with_curve(curve)

    def test_no_warn_without_status(self):
        """No Status -> no warning, even when curve includes (0, 0)."""
        curve = PiecewiseConversion(points={'fuel': [0, 50, 100], 'Heat': [0, 45, 70]})
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            self._build_with_curve(curve)

    def test_no_warn_when_only_one_flow_zero(self):
        """Only one flow is zero at a breakpoint (not all) -> no warning."""
        # heat=0 at first bp but fuel=10 -> the curve doesn't include the origin.
        curve = PiecewiseConversion(
            points={'fuel': [10, 50, 100], 'Heat': [0, 45, 70]},
            status=Status(effects_per_startup={'cost': 1}),
        )
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            self._build_with_curve(curve)
