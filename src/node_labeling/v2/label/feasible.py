import math
import numpy as np

from dataclasses import dataclass
from typing import Callable

from node_labeling.v2.filter.shape_rules import LabelRules
from node_labeling.v2.label.slider import Slider
from node_labeling.v2.utils.intervals import IntervalSet, intervals_from_mask

Predicate = Callable[[np.ndarray, np.ndarray], np.ndarray]

@dataclass
class Feasible:
    '''
    Positions of a label on its slider that pass the fixed rules.

    Attributes
    ----------
    intervals : IntervalSet
        arc lengths that pass the hard rules (`LabelRules.base_ok`)
    preferred : IntervalSet
        the part of `intervals` that also passes the neighbor rule
    '''
    intervals: IntervalSet
    preferred: IntervalSet

def feasible_set(slider: Slider, rules: LabelRules, step: float, tol: float) -> Feasible:
    '''
    Feasible positions of a label: sample the slider every `step`, then bisect
    every interval end to `tol`. Interval ends are always tested positions.

    Parameters
    ----------
    slider : Slider
        slider of the label
    rules : LabelRules
        fixed rules of the label
    step : float
        sampling distance along the slider
    tol : float
        bisection tolerance

    Returns
    -------
    feasible : Feasible
        feasible and preferred interval sets
    '''
    base = _intervals(slider, rules.base_ok, step, tol)
    if base.is_empty:
        return Feasible(base, base)
    return Feasible(base, base.intersection(_intervals(slider, rules.neighbor_ok, step, tol)))

def _intervals(slider: Slider, predicate: Predicate, step: float, tol: float) -> IntervalSet:
    '''
    Parameters
    ----------
    slider : Slider
        slider
    predicate : Predicate
        vectorized test (t, positions) -> bool array
    step : float
        sampling distance
    tol : float
        bisection tolerance

    Returns
    -------
    intervals : IntervalSet
        arc lengths where the predicate holds
    '''
    length = slider.length
    n = max(2, math.ceil(length / step) + 1)
    t = np.linspace(0.0, length, n)
    ok = predicate(t, slider.positions(t))

    def test(x: float) -> bool:
        return bool(predicate(np.array([x]), slider.positions(x))[0])

    def bisect(good: float, bad: float) -> float:
        while abs(bad - good) > tol:
            mid = (good + bad) / 2.0
            if test(mid):
                good = mid
            else:
                bad = mid
        return good

    out = []
    for i, j in intervals_from_mask(ok):
        a = t[i] if i == 0 else bisect(t[i], t[i - 1])
        b = t[j] if j == n - 1 else bisect(t[j], t[j + 1])
        out.append((a, b))
    return IntervalSet(out)
