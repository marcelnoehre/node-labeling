import math
import numpy as np

from typing import Iterable, Iterator, List, Tuple

class IntervalSet:
    '''
    Finite union of disjoint closed intervals [a, b], sorted.
    '''
    def __init__(self, intervals: Iterable[Tuple[float, float]] = ()):
        merged: List[Tuple[float, float]] = []
        for a, b in sorted((float(a), float(b)) for a, b in intervals if b >= a):
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        self._iv = merged

    def __iter__(self) -> Iterator[Tuple[float, float]]:
        return iter(self._iv)

    def __len__(self) -> int:
        return len(self._iv)

    def __repr__(self) -> str:
        return 'IntervalSet(' + ', '.join(f'[{a:.4f}, {b:.4f}]' for a, b in self._iv) + ')'

    @property
    def is_empty(self) -> bool:
        '''
        Returns
        -------
        empty : bool
            whether the set has no interval
        '''
        return not self._iv

    @property
    def measure(self) -> float:
        '''
        Returns
        -------
        measure : float
            total length of the intervals
        '''
        return sum(b - a for a, b in self._iv)

    def contains(self, t: float, eps: float = 0.0) -> bool:
        '''
        Parameters
        ----------
        t : float
            value
        eps : float
            tolerance

        Returns
        -------
        contained : bool
            whether t lies in one of the intervals
        '''
        return any(a - eps <= t <= b + eps for a, b in self._iv)

    def intersection(self, other: 'IntervalSet') -> 'IntervalSet':
        '''
        Parameters
        ----------
        other : IntervalSet
            second set

        Returns
        -------
        intersection : IntervalSet
            values in both sets
        '''
        out, i, j = [], 0, 0
        A, B = self._iv, other._iv
        while i < len(A) and j < len(B):
            a, b = max(A[i][0], B[j][0]), min(A[i][1], B[j][1])
            if a <= b:
                out.append((a, b))
            if A[i][1] < B[j][1]:
                i += 1
            else:
                j += 1
        return IntervalSet(out)

    def sample(self, step: float) -> np.ndarray:
        '''
        Sample values: both ends of every interval and evenly spaced values in
        between, at most `step` apart.

        Parameters
        ----------
        step : float
            maximal spacing

        Returns
        -------
        t : np.ndarray
            sorted sample values
        '''
        parts = []
        for a, b in self._iv:
            n = max(1, math.ceil((b - a) / step)) if b > a else 0
            parts.append(np.linspace(a, b, n + 1))
        return np.concatenate(parts) if parts else np.empty(0)

def intervals_from_mask(ok: np.ndarray) -> List[Tuple[int, int]]:
    '''
    Index ranges of the runs of True in a boolean mask.

    Parameters
    ----------
    ok : np.ndarray
        boolean mask

    Returns
    -------
    runs : List[Tuple[int, int]]
        (first, last) index of every run
    '''
    runs, start = [], None
    for i, v in enumerate(ok):
        if v and start is None:
            start = i
        if not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(ok) - 1))
    return runs
