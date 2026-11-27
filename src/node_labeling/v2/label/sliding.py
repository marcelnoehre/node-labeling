import numpy as np
import networkx as nx
import shapely

from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List, Optional, Tuple

from shapely.geometry import LineString
from shapely.strtree import STRtree

from node_labeling.models.label_type import LabelType
from node_labeling.utils.config import Config
from node_labeling.utils.constants import (
    EDGE_CLEARANCE_MM, LINE_WIDTH, MM_PER_UNIT, NODE_GAP_MM, NODE_SIZE, PADDING_MM, PLACEMENT_STEP_MM,
    SLIDER_STEP_MM, SLIDER_TOL_MM
)
from node_labeling.v2.filter.shape_rules import LabelRules, Obstacles
from node_labeling.v2.label.feasible import Feasible, feasible_set
from node_labeling.v2.label.slider import Slider
from node_labeling.v2.models.candidate import Candidate
from node_labeling.v2.models.label_shape import LabelShape
from node_labeling.v2.utils.intervals import IntervalSet
from node_labeling.v2.utils.shapes import translate_many

CLEARANCE_CAP: float = 2.0

@dataclass(eq=False)
class SlidingLabel:
    '''
    A label with its slider and its feasible positions.

    Attributes
    ----------
    label_id : int
        label id
    node_id : Hashable
        node of the label
    label_type : LabelType
        label type
    text : str
        formatted LaTeX label
    shape : LabelShape
        label shape
    slider : Slider
        slider around the node
    rules : LabelRules
        fixed rules
    feasible : Feasible
        feasible positions on the slider
    '''
    label_id: int
    node_id: Hashable
    label_type: LabelType
    text: str
    shape: LabelShape
    slider: Slider
    rules: LabelRules
    feasible: Feasible

def prepare_labels(
        G: nx.Graph,
        labels: Iterable[Tuple[int, Hashable, LabelType, str]],
        obstacles: Obstacles,
        cfg: Config
) -> List[SlidingLabel]:
    '''
    Measure every label, build its slider and its feasible set. The node
    center keeps at least the marker radius + `NODE_GAP_MM` from the letters
    (slider node radius), so the node marker never touches the label.

    Parameters
    ----------
    G : nx.Graph
        graph containing normalized positions
    labels : Iterable[Tuple[int, Hashable, LabelType, str]]
        (label id, node id, label type, formatted text)
    obstacles : Obstacles
        fixed obstacles
    cfg : Config
        configuration

    Returns
    -------
    labels : List[SlidingLabel]
        labels with sliders and feasible sets
    '''
    marker_mm = (np.sqrt(NODE_SIZE) / 2.0 + LINE_WIDTH / 2.0) * 25.4 / 72.0
    out = []
    for lid, nid, label_type, text in labels:
        shape = LabelShape.from_tex(text, cfg)
        node_radius = max(0.0, (marker_mm + NODE_GAP_MM) / MM_PER_UNIT - shape.padding)
        slider = Slider.build(G.nodes[nid]['pos'], shape, label_type, node_radius)
        clearance = shape.padding * EDGE_CLEARANCE_MM / PADDING_MM
        rules = LabelRules(obstacles, nid, shape, label_type, slider, clearance)
        feasible = feasible_set(slider, rules, SLIDER_STEP_MM / MM_PER_UNIT, SLIDER_TOL_MM / MM_PER_UNIT)
        out.append(SlidingLabel(lid, nid, label_type, text, shape, slider, rules, feasible))
    return out

def place_labels(labels: List[SlidingLabel], obstacles: Obstacles) -> Dict[int, Optional[Candidate]]:
    '''
    Choose one position per label on its feasible set, so that no two labels
    come closer than `LABEL_DISTANCE` paddings (spacers do not overlap;
    continuous version of the v1 rules L1–L3):

    1. a label with a position that no other unplaced label can reach is placed
       there (largest clearance, then the middle of the free stretch). Labels
       with preferred positions left (neighbor rule) only take free preferred ones.
    2. otherwise the most constrained label (fewest feasible positions) is placed
       where it removes the fewest positions from the others: first by the
       number of labels it would leave without any position, then preferred
       before not preferred, then by the removed length (ties: clearance, then
       middle). So the neighbor rule never costs a label its place.
    3. positions of other labels whose spacer meets the placed spacer are removed;
       labels without positions become overflow labels

    The feasible sets are sampled every `PLACEMENT_STEP_MM` (interval ends
    included); spacer overlaps are tested exactly between these samples.

    Parameters
    ----------
    labels : List[SlidingLabel]
        labels with feasible sets
    obstacles : Obstacles
        fixed obstacles (for the clearance)

    Returns
    -------
    placed : Dict[int, Optional[Candidate]]
        placed candidate per label id, None for overflow labels
    '''
    step = PLACEMENT_STEP_MM / MM_PER_UNIT
    K = len(labels)
    ts, pos, run_id, prefs = [], [], [], []
    for L in labels:
        t, rid = _sample(L.feasible, step)
        P = L.slider.positions(t) if len(t) else np.empty((0, 2))
        pref = np.zeros(len(t), dtype=bool)
        if len(t):
            ok = L.rules.base_ok(t, P)
            t, P, rid = t[ok], P[ok], rid[ok]
            pref = L.rules.neighbor_ok(t, P)
        ts.append(t)
        pos.append(P)
        run_id.append(rid)
        prefs.append(pref)

    counts = np.array([len(t) for t in ts], dtype=int)
    start = np.concatenate([[0], np.cumsum(counts)])
    N = int(start[-1])
    owner = np.repeat(np.arange(K), counts)
    preferred = np.concatenate(prefs) if N else np.empty(0, dtype=bool)
    spacers = np.concatenate([translate_many(L.shape.spacer, P) for L, P in zip(labels, pos)]) if N else np.empty(0, dtype=object)

    if N:
        a, b = STRtree(spacers).query(spacers, predicate='intersects')
        keep = owner[a] != owner[b]
        a, b = a[keep], b[keep]
        order = np.argsort(a, kind='stable')
        a, b = a[order], b[order]
    else:
        a = b = np.empty(0, dtype=int)
    indptr = np.searchsorted(a, np.arange(N + 1))

    alive = np.ones(N, dtype=bool)
    alive_cnt = counts.copy()
    nbr_alive = np.diff(indptr).astype(int)
    chosen: Dict[int, int] = {}
    unplaced = {k for k in range(K) if counts[k] > 0}

    def nbrs(g: int) -> np.ndarray:
        return b[indptr[g]:indptr[g + 1]]

    def kill(hs: np.ndarray):
        for h in hs:
            if alive[h]:
                alive[h] = False
                alive_cnt[owner[h]] -= 1
                nbr_alive[nbrs(h)] -= 1

    def place(k: int, g: int):
        chosen[k] = g
        unplaced.discard(k)
        kill(nbrs(g))
        kill(np.arange(start[k], start[k + 1]))
        for j in [j for j in unplaced if alive_cnt[j] == 0]:
            unplaced.discard(j)

    def best(k: int, gs: np.ndarray) -> int:
        others = [(labels[j], pos[j][chosen[j] - start[j]]) for j in chosen]
        clear = _clearance(labels[k], pos[k][gs - start[k]], obstacles, others)
        middle = _middle(ts[k], run_id[k], gs - start[k])
        return int(gs[np.lexsort((-middle, -clear, ~preferred[gs]))[0]])

    while unplaced:
        progress = True
        while progress:
            progress = False
            for k in sorted(unplaced):
                if k not in unplaced:
                    continue
                idx = np.arange(start[k], start[k + 1])
                idx = idx[alive[idx]]
                if preferred[idx].any():
                    idx = idx[preferred[idx]]
                free = idx[nbr_alive[idx] == 0]
                if len(free):
                    place(k, best(k, free))
                    progress = True
        if not unplaced:
            break

        k = min(unplaced, key=lambda j: (alive_cnt[j], labels[j].label_id))
        idx = np.arange(start[k], start[k + 1])
        cand = idx[alive[idx]]
        loss = np.empty((len(cand), 3))
        for c, g in enumerate(cand):
            nb = nbrs(g)
            nb = nb[alive[nb]]
            removed = np.bincount(owner[nb], minlength=K)
            hit = removed > 0
            loss[c] = (np.count_nonzero(removed[hit] == alive_cnt[hit]), not preferred[g], removed.sum())
        top = np.ones(len(cand), dtype=bool)
        for col in range(3):
            top &= loss[:, col] == loss[top, col].min()
        place(k, best(k, cand[top]))

    placed: Dict[int, Optional[Candidate]] = {}
    for k, L in enumerate(labels):
        if k not in chosen:
            placed[L.label_id] = None
            continue
        x, y = pos[k][chosen[k] - start[k]]
        placed[L.label_id] = Candidate(
            label_id=L.label_id, node_id=L.node_id, node_pos=L.slider.node, label_type=L.label_type,
            text=L.text, shape=L.shape, position=(float(x), float(y))
        )
    return placed

def _sample(feasible: Feasible, step: float) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Parameters
    ----------
    feasible : Feasible
        feasible set
    step : float
        maximal spacing

    Returns
    -------
    t : np.ndarray
        samples (interval ends included)
    run_id : np.ndarray
        index of the interval of every sample
    '''
    ts, ids = [], []
    for i, (lo, hi) in enumerate(feasible.intervals):
        t = IntervalSet([(lo, hi)]).sample(step)
        ts.append(t)
        ids.append(np.full(len(t), i))
    if not ts:
        return np.empty(0), np.empty(0, dtype=int)
    return np.concatenate(ts), np.concatenate(ids)

def _middle(t: np.ndarray, run_id: np.ndarray, sel: np.ndarray) -> np.ndarray:
    '''
    Distance of every selected sample to the nearer end of its stretch: a
    maximal run of consecutive selected samples in the same interval.

    Parameters
    ----------
    t : np.ndarray
        all samples of the label
    run_id : np.ndarray
        interval index of every sample
    sel : np.ndarray
        sorted indices of the selected samples

    Returns
    -------
    middle : np.ndarray
        distance to the nearer stretch end, per selected sample
    '''
    out = np.zeros(len(sel))
    s = 0
    for e in range(1, len(sel) + 1):
        if e == len(sel) or sel[e] != sel[e - 1] + 1 or run_id[sel[e]] != run_id[sel[e - 1]]:
            lo, hi = t[sel[s]], t[sel[e - 1]]
            out[s:e] = np.minimum(t[sel[s:e]] - lo, hi - t[sel[s:e]])
            s = e
    return out

def _clearance(
        label: SlidingLabel,
        positions: np.ndarray,
        obstacles: Obstacles,
        placed: List[Tuple[SlidingLabel, np.ndarray]]
) -> np.ndarray:
    '''
    Distance from the ink to the nearest edge, other node or placed label, capped
    at `CLEARANCE_CAP` paddings. Edges of the label's own node are measured from
    `CLEARANCE_CAP` paddings away from the node on, since they all start at the node.

    Parameters
    ----------
    label : SlidingLabel
        label
    positions : np.ndarray
        (n, 2) drawing points
    obstacles : Obstacles
        fixed obstacles
    placed : List[Tuple[SlidingLabel, np.ndarray]]
        labels placed so far with their positions

    Returns
    -------
    clearance : np.ndarray
        capped clearance per position
    '''
    p = label.shape.padding
    cap = CLEARANCE_CAP * p
    x0, y0, x1, y1 = label.shape.pad.bounds
    lo = positions.min(axis=0) + (x0 - cap, y0 - cap)
    hi = positions.max(axis=0) + (x1 + cap, y1 + cap)
    window = shapely.box(lo[0], lo[1], hi[0], hi[1])
    node = np.asarray(label.slider.node)

    obst = []
    for e in obstacles.edge_tree.query(window):
        u, v = obstacles.edge_ends[e]
        if label.node_id not in (u, v):
            obst.append(obstacles.edges[e])
            continue
        far = np.asarray(obstacles.edges[e].coords)[1 if u == label.node_id else 0]
        d = np.hypot(*(far - node))
        if d > cap:
            obst.append(LineString([node + (far - node) * cap / d, far]))
    near = ((obstacles.node_xy >= lo) & (obstacles.node_xy <= hi)).all(axis=1)
    for i in np.flatnonzero(near):
        if obstacles.node_ids[i] != label.node_id:
            obst.append(shapely.points(obstacles.node_xy[i]))
    for other, P in placed:
        ox0, oy0, ox1, oy1 = other.shape.ink.bounds
        if P[0] + ox1 >= lo[0] and P[0] + ox0 <= hi[0] and P[1] + oy1 >= lo[1] and P[1] + oy0 <= hi[1]:
            obst.append(translate_many(other.shape.ink, P[None])[0])
    if not obst:
        return np.full(len(positions), cap)

    obst = np.array(obst, dtype=object)
    pads = translate_many(label.shape.pad, positions)
    d = shapely.distance(pads[:, None], obst[None, :]) + p
    inside = d <= p
    if inside.any():
        i, j = np.nonzero(inside)
        inks = translate_many(label.shape.ink, positions[i])
        d[i, j] = shapely.distance(inks, obst[j])
    return np.minimum(d.min(axis=1), cap)
