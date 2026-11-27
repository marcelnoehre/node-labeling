import warnings
import numpy as np
import shapely

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from node_labeling.models.label_type import LabelType
from node_labeling.utils.config import Config
from node_labeling.utils.constants import DOT_CLEARANCE_MM, EDGE_CLEARANCE_MM, FOREIGN_NODE_MM, LINE_WIDTH, MM_PER_UNIT, NODE_GAP_MM, NODE_SIZE, PADDING_MM
from node_labeling.v2.filter.shape_rules import clearance_shape
from node_labeling.v2.models.candidate import Candidate
from node_labeling.v2.models.label_shape import LabelShape
from node_labeling.v2.overflow.binders import DOT_RADIUS, MARKER_RADIUS, Layout, attach_many, route, straighten
from node_labeling.v2.utils.shapes import translate_many

FREE_STEP_MM: float = 1.0
REFINE_STEP_MM: float = 0.25
BALANCE_CAP_MM: float = 5.0
BALANCE_FRACTIONS: Tuple[float, ...] = (0.0, 0.5, 1.0)
MAX_POSITION_TRIES: int = 10
MAX_ROUNDS: int = 5
DEAD_RADIUS_MM: float = 1.0
ATTACH_SLACK_MM: float = 3.0
TIER: float = 1e6
RAY_CHUNK: int = 256
TOP_SPOTS: int = 400
BALANCE_RADIUS_MM: float = 3.0
BALANCE_STEP_MM: float = 0.5
BALANCE_PASSES: int = 2

def place_overflow(
        layout: Layout,
        overflow: Dict[int, Candidate],
        cfg: Config
) -> Tuple[Dict[int, Candidate], Dict[int, List[Tuple[Candidate, float]]]]:
    '''
    Place the labels without a position at their node in the free space around
    their node (T9), then draw their binders.

    1. positions: drawing points on a `FREE_STEP_MM` grid within
       `cfg.overflow_radius_mm` of the node (doubled once if nothing fits),
       inside or outside the diagram, with the rules of the labels at their
       nodes: letters `EDGE_CLEARANCE_MM` from edges, `LABEL_DISTANCE` paddings
       to other labels, the own node marker + `NODE_GAP_MM` from the letters;
       other nodes at least `FOREIGN_NODE_MM` from the letters (Q9.6), so the
       label is not read as theirs
    2. quality, from the position alone (Q9.0): the half plane first (attribute
       labels above their node, object labels below; the other half plane only if
       nothing is left in the own one), then cost = closeness to the node +
       `cfg.w_overflow_balance` · crowding + `cfg.overflow_inside_mm` inside the
       diagram (all in mm). Crowding = Σ over the four sides of
       (`BALANCE_CAP_MM` − free space)² / `BALANCE_CAP_MM` for sides with less
       free space (to edges, other letters, node markers): a label moves away
       from a neighbour that is too close and sits in the middle of a tight gap,
       but is not pulled away from its node into open space
    3. choice: every label takes its best position; where two choices collide, the
       label that would lose more by moving keeps its position (one per round)
    4. refinement: `BALANCE_PASSES` passes within `BALANCE_RADIUS_MM` on a
       `BALANCE_STEP_MM` grid in which the other overflow labels count for the
       balance too (so neighbours share the space), then one pass within one grid
       step on a `REFINE_STEP_MM` grid
    5. binders: straight if conflict-free (the dot may slide along its ring by up
       to `ATTACH_SLACK_MM` of binder length, e.g. to keep clear of an edge), else
       a Bézier curve (`route`), no binder crossings. If a label gets no acceptable
       binder, its position (and the positions within `DEAD_RADIUS_MM`) is dropped
       and steps 3–5 run again for all labels (at most `MAX_ROUNDS`); labels still
       without binder then try their next best positions one by one (Q9.5,
       `MAX_POSITION_TRIES`). Final straighten pass.

    The dot room is part of a position (step 1): the dot at the attachment
    nearest to the node keeps `DOT_CLEARANCE_MM` from edges, node markers and
    letters.

    Parameters
    ----------
    layout : Layout
        fixed context (labels at their nodes, obstacles)
    overflow : Dict[int, Candidate]
        labels without position (at their node, no binder)
    cfg : Config
        configuration

    Returns
    -------
    overflow : Dict[int, Candidate]
        overflow labels with position and binder (binder None if nothing fits)
    spots : Dict[int, List[Tuple[Candidate, float]]]
        candidate positions with their cost per label (dev plots)
    '''
    context = _Context(layout, cfg)
    lids = sorted(overflow)
    spots: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for lid in lids:
        P, key = context.spots(overflow[lid], cfg.overflow_radius_mm)
        if not len(P):
            P, key = context.spots(overflow[lid], 2 * cfg.overflow_radius_mm)
        spots[lid] = (P, key)

    dead = {lid: np.zeros(len(spots[lid][0]), dtype=bool) for lid in lids}
    for round_ in range(MAX_ROUNDS):
        alive = {lid: np.flatnonzero(~dead[lid]) for lid in lids}
        sub = _choose(overflow, {lid: (spots[lid][0][alive[lid]], spots[lid][1][alive[lid]]) for lid in lids})
        chosen = {lid: int(alive[lid][i]) for lid, i in sub.items()}
        order = sorted(chosen, key=lambda lid: spots[lid][1][chosen[lid]])
        placed: Dict[int, Candidate] = {lid: overflow[lid].moved(spots[lid][0][chosen[lid]]) for lid in order}
        for radius, step in [(BALANCE_RADIUS_MM, BALANCE_STEP_MM)] * BALANCE_PASSES + [(FREE_STEP_MM, REFINE_STEP_MM)]:
            for lid in order:
                others = [c for o, c in placed.items() if o != lid]
                placed[lid] = overflow[lid].moved(context.refine(placed[lid], others, radius, step))
        result = dict(overflow)
        failed = []
        for lid in order:
            r = route(layout, lid, placed[lid], result, cfg, soft=True, attachment_slack=ATTACH_SLACK_MM / MM_PER_UNIT,
                      allow_binder_crossings=False)
            if r is None:
                failed.append(lid)
            else:
                result[lid] = placed[lid].moved(placed[lid].position, r[0])
        if not failed or round_ == MAX_ROUNDS - 1:
            break
        for lid in failed:
            P = spots[lid][0]
            dead[lid] |= np.hypot(*(P - P[chosen[lid]]).T) <= DEAD_RADIUS_MM / MM_PER_UNIT

    for lid in order:
        if result[lid].binder is None:
            result[lid] = _with_binder(layout, lid, placed[lid], spots[lid], result, placed, context, cfg)
    for lid in lids:
        if lid not in chosen:
            warnings.warn(f'Label {lid} ({overflow[lid].text!r}) could not be placed and is drawn at its node')
    result = straighten(layout, result, [], cfg)

    dev = {lid: [(overflow[lid].moved(p), float(k)) for p, k in zip(*spots[lid])] for lid in lids}
    return result, dev

class _Context:
    '''
    Obstacles of the free-space search, built once: spacers of the labels at
    their nodes, edges, node markers and the hulls of their letters for the
    balance rays.
    '''
    def __init__(self, layout: Layout, cfg: Config):
        self.layout = layout
        self.cfg = cfg
        direct = layout.direct
        self.spacers = [shapely.affinity.translate(c.shape.spacer, *c.position) for c in direct]
        shapely.prepare(self.spacers)
        self.spacer_tree = STRtree(self.spacers)
        obst = layout.obstacles
        self.hull = obst.hull
        self.marker = (np.sqrt(NODE_SIZE) / 2.0 + LINE_WIDTH / 2.0) * 25.4 / 72.0 / MM_PER_UNIT
        self.keep = self.marker + NODE_GAP_MM / MM_PER_UNIT
        segs = np.array([np.asarray(e.coords) for e in obst.edges], dtype=float).reshape(-1, 2, 2)
        self.segs = np.concatenate([segs] + [_hull_segments(c) for c in direct])

    def spots(self, candidate: Candidate, radius_mm: float) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Parameters
        ----------
        candidate : Candidate
            overflow label (at its node)
        radius_mm : float
            search radius around the node

        Returns
        -------
        positions : np.ndarray
            (n, 2) feasible drawing points
        keys : np.ndarray
            tier · `TIER` + cost per position
        '''
        step = FREE_STEP_MM / MM_PER_UNIT
        r = radius_mm / MM_PER_UNIT
        node = np.asarray(candidate.node_pos)
        xs = np.arange(-r, r + 1e-9, step)
        offsets = np.stack(np.meshgrid(xs, xs), axis=-1).reshape(-1, 2)
        P = node + offsets[np.hypot(*offsets.T) <= r]
        P = P[self.feasible(candidate, P)]
        P = P[self.dot_room(candidate, P)]
        return P, self.keys(candidate, P)

    def dot_room(self, candidate: Candidate, P: np.ndarray) -> np.ndarray:
        '''
        Whether the binder dot at the attachment nearest to the node keeps
        `DOT_CLEARANCE_MM` from edges, node markers and the letters of the labels
        at their nodes (the dot belongs to the label's footprint).

        Parameters
        ----------
        candidate : Candidate
            overflow label
        P : np.ndarray
            (n, 2) drawing points

        Returns
        -------
        ok : np.ndarray
            whether the dot has room at each position
        '''
        if not len(P):
            return np.ones(0, dtype=bool)
        ends = np.array([b.end for b in attach_many([candidate.moved(p) for p in P])])
        clear = DOT_CLEARANCE_MM / MM_PER_UNIT
        discs = shapely.buffer(shapely.points(ends), DOT_RADIUS + clear)
        ok = np.ones(len(P), dtype=bool)
        ok[self.layout.obstacles.edge_tree.query(discs, predicate='intersects')[0]] = False
        ok[self.layout.ink_tree.query(discs, predicate='intersects')[0]] = False
        nodes = self.layout.obstacles.node_xy
        near = np.hypot(ends[:, None, 0] - nodes[None, :, 0], ends[:, None, 1] - nodes[None, :, 1]) < MARKER_RADIUS + DOT_RADIUS + clear
        ok &= ~near.any(axis=1)
        return ok

    def feasible(self, candidate: Candidate, P: np.ndarray, others: Optional[List[Candidate]] = None) -> np.ndarray:
        '''
        Parameters
        ----------
        candidate : Candidate
            overflow label
        P : np.ndarray
            (n, 2) drawing points
        others : Optional[List[Candidate]]
            placed overflow labels to keep the label distance to

        Returns
        -------
        ok : np.ndarray
            whether the label fits at each position
        '''
        shape = candidate.shape
        ok = np.ones(len(P), dtype=bool)
        if not len(P):
            return ok
        clear = clearance_shape(shape, shape.padding * EDGE_CLEARANCE_MM / PADDING_MM)
        hits = self.layout.obstacles.edge_tree.query(translate_many(clear, P), predicate='intersects')
        ok[hits[0]] = False
        idx = np.flatnonzero(ok)
        if len(idx) and self.spacers:
            hits = self.spacer_tree.query(translate_many(shape.spacer, P[idx]), predicate='intersects')
            ok[idx[hits[0]]] = False
        idx = np.flatnonzero(ok)
        if len(idx):
            nodes = self.layout.obstacles.node_xy
            own = np.hypot(*(nodes - np.asarray(candidate.node_pos)).T) <= 1e-9
            for mask, keep in ((own, self.keep), (~own, FOREIGN_NODE_MM / MM_PER_UNIT)):
                if not mask.any() or not len(idx):
                    continue
                rel_x = nodes[None, mask, 0] - P[idx, None, 0]
                rel_y = nodes[None, mask, 1] - P[idx, None, 1]
                ok[idx[shapely.intersects_xy(_node_keep(shape, keep), rel_x, rel_y).any(axis=1)]] = False
                idx = np.flatnonzero(ok)
        idx = np.flatnonzero(ok)
        if len(idx) and others:
            spacers = translate_many(shape.spacer, P[idx])
            for other in others:
                ok[idx[shapely.intersects(spacers, shapely.affinity.translate(other.shape.spacer, *other.position))]] = False
        return ok

    def keys(self, candidate: Candidate, P: np.ndarray, others: Optional[List[Candidate]] = None) -> np.ndarray:
        '''
        Quality of positions (Q9.0): tier · `TIER` + cost, see `place_overflow`.

        Parameters
        ----------
        candidate : Candidate
            overflow label
        P : np.ndarray
            (n, 2) drawing points
        others : Optional[List[Candidate]]
            placed overflow labels, counted for the balance

        Returns
        -------
        keys : np.ndarray
            key per position (smaller is better)
        '''
        if not len(P):
            return np.empty(0)
        shape = candidate.shape
        node = np.asarray(candidate.node_pos)
        x0, y0, x1, y1 = shape.ink.bounds
        tier = np.zeros(len(P))
        if candidate.label_type == LabelType.INTENT:
            tier = (P[:, 1] + y0 < node[1] - 1e-9).astype(float)
        elif candidate.label_type == LabelType.EXTENT:
            tier = (P[:, 1] + y1 > node[1] + 1e-9).astype(float)
        close = shapely.distance(shape.ink, shapely.points(node - P)) * MM_PER_UNIT
        segs = np.concatenate([self.segs] + [_hull_segments(o) for o in others or []])
        nodes = self.layout.obstacles.node_xy
        nodes = nodes[np.hypot(*(nodes - node).T) > 1e-9]
        origins, dirs, sides = _balance_rays(x0, y0, x1, y1)
        balance = np.empty(len(P))
        for s in range(0, len(P), RAY_CHUNK):
            Q = P[s:s + RAY_CHUNK]
            free = _ray(Q[:, None, :] + origins[None], dirs, segs, nodes, self.marker)
            crowd = BALANCE_CAP_MM - np.stack([free[:, sides == k].min(axis=1) for k in range(4)], axis=1) * MM_PER_UNIT
            balance[s:s + RAY_CHUNK] = (np.clip(crowd, 0.0, None) ** 2).sum(axis=1) / BALANCE_CAP_MM
        cost = close + self.cfg.w_overflow_balance * balance
        if self.hull is not None:
            inside = shapely.contains_xy(self.hull, P[:, 0] + (x0 + x1) / 2.0, P[:, 1] + (y0 + y1) / 2.0)
            cost = cost + inside * self.cfg.overflow_inside_mm
        return tier * TIER + cost

    def refine(self, candidate: Candidate, others: List[Candidate], radius_mm: float, step_mm: float) -> Tuple[float, float]:
        '''
        Best position on a `step_mm` grid within `radius_mm` of the current one:
        feasible, label distance to the other overflow labels, dot room; the other
        overflow labels count for the balance.

        Parameters
        ----------
        candidate : Candidate
            overflow label at its current position
        others : List[Candidate]
            the other overflow labels at their current positions
        radius_mm : float
            search radius
        step_mm : float
            grid step

        Returns
        -------
        position : Tuple[float, float]
            refined drawing point
        '''
        p = np.asarray(candidate.position)
        r = radius_mm / MM_PER_UNIT
        xs = np.arange(-r, r + 1e-9, step_mm / MM_PER_UNIT)
        P = p + np.stack(np.meshgrid(xs, xs), axis=-1).reshape(-1, 2)
        P = P[self.feasible(candidate, P, others)]
        P = P[self.dot_room(candidate, P)]
        if not len(P):
            return float(p[0]), float(p[1])
        keys = self.keys(candidate, P, others)
        current = self.keys(candidate, p[None], others)[0]
        i = int(np.argmin(keys))
        if keys[i] >= current:
            return float(p[0]), float(p[1])
        return float(P[i, 0]), float(P[i, 1])

def _hull_segments(candidate: Candidate) -> np.ndarray:
    '''
    Parameters
    ----------
    candidate : Candidate
        placed label

    Returns
    -------
    segments : np.ndarray
        (m, 2, 2) edges of the convex hull of its letters (balance obstacles)
    '''
    ring = np.asarray(candidate.ink.convex_hull.exterior.coords)
    return np.stack([ring[:-1], ring[1:]], axis=1)

def _balance_rays(x0: float, y0: float, x1: float, y1: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Rays that measure the free space on the four sides of a label (letter bounds
    x0..x1, y0..y1): per side, straight rays from both ends and the middle of the
    side, and diagonal rays from its two corners (so a neighbour beyond a corner
    counts too).

    Returns
    -------
    origins : np.ndarray
        (k, 2) ray origins relative to the drawing point
    directions : np.ndarray
        (k, 2) unit directions
    sides : np.ndarray
        (k,) side of each ray: 0 left, 1 right, 2 down, 3 up
    '''
    d = 1.0 / np.sqrt(2.0)
    origins, dirs, sides = [], [], []
    for f in BALANCE_FRACTIONS:
        y = y0 + f * (y1 - y0)
        x = x0 + f * (x1 - x0)
        origins += [(x0, y), (x1, y), (x, y0), (x, y1)]
        dirs += [(-1, 0), (1, 0), (0, -1), (0, 1)]
        sides += [0, 1, 2, 3]
    for (cx, cy), (dx, dy), (sx, sy) in [((x0, y0), (-d, -d), (0, 2)), ((x1, y0), (d, -d), (1, 2)),
                                         ((x0, y1), (-d, d), (0, 3)), ((x1, y1), (d, d), (1, 3))]:
        origins += [(cx, cy), (cx, cy)]
        dirs += [(dx, dy), (dx, dy)]
        sides += [sx, sy]
    return np.array(origins, dtype=float), np.array(dirs, dtype=float), np.array(sides)

def _ray(O: np.ndarray, D: np.ndarray, segs: np.ndarray, nodes: np.ndarray, marker: float) -> np.ndarray:
    '''
    Free distance along rays to the nearest segment or node marker, capped at
    `BALANCE_CAP_MM`.

    Parameters
    ----------
    O : np.ndarray
        (n, k, 2) ray origins
    D : np.ndarray
        (k, 2) unit directions
    segs : np.ndarray
        (m, 2, 2) obstacle segments
    nodes : np.ndarray
        (j, 2) node marker centres
    marker : float
        node marker radius

    Returns
    -------
    free : np.ndarray
        (n, k) distances
    '''
    cap = BALANCE_CAP_MM / MM_PER_UNIT
    free = np.full(O.shape[:2], cap)
    if len(segs):
        A = segs[:, 0]
        E = segs[:, 1] - A
        denom = D[:, None, 0] * E[None, :, 1] - D[:, None, 1] * E[None, :, 0]
        AO = A[None, None] - O[:, :, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (AO[..., 0] * E[None, None, :, 1] - AO[..., 1] * E[None, None, :, 0]) / denom[None]
            u = (AO[..., 0] * D[None, :, None, 1] - AO[..., 1] * D[None, :, None, 0]) / denom[None]
        hit = (np.abs(denom[None]) > 1e-12) & (t >= 0) & (u >= 0) & (u <= 1)
        free = np.minimum(free, np.where(hit, t, np.inf).min(axis=-1))
    if len(nodes):
        OC = O[:, :, None] - nodes[None, None]
        b = (OC * D[None, :, None]).sum(axis=-1)
        c = (OC ** 2).sum(axis=-1) - marker ** 2
        disc = b ** 2 - c
        t = -b - np.sqrt(np.clip(disc, 0.0, None))
        hit = (disc >= 0) & (t >= -marker)
        free = np.minimum(free, np.where(hit, np.clip(t, 0.0, None), np.inf).min(axis=-1))
    return free

def _choose(overflow: Dict[int, Candidate], spots: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Dict[int, int]:
    '''
    Choose one position per label. Every label points at its best position left;
    if two of these collide (spacers overlap), the label that would lose most by
    taking its best position compatible with the others keeps its choice, and the
    positions of other labels colliding with it are removed. Repeated until the
    choices are compatible. Only the `TOP_SPOTS` best positions per label take part.

    Parameters
    ----------
    overflow : Dict[int, Candidate]
        overflow labels
    spots : Dict[int, Tuple[np.ndarray, np.ndarray]]
        positions and keys per label

    Returns
    -------
    chosen : Dict[int, int]
        index of the chosen position per label (labels without position are missing)
    '''
    lids = [lid for lid in sorted(spots) if len(spots[lid][0])]
    if not lids:
        return {}
    top = {lid: np.argsort(spots[lid][1], kind='stable')[:TOP_SPOTS] for lid in lids}
    per_label = [translate_many(overflow[lid].shape.spacer, spots[lid][0][top[lid]]) for lid in lids]
    spacers = np.concatenate(per_label)
    owner = np.concatenate([np.full(len(top[lid]), i) for i, lid in enumerate(lids)])
    start = np.concatenate([[0], np.cumsum([len(top[lid]) for lid in lids])])
    keys = np.concatenate([spots[lid][1][top[lid]] for lid in lids])
    a_parts, b_parts = [], []
    for i in range(len(lids)):
        others = np.flatnonzero(owner != i)
        if not len(others):
            continue
        q, t = STRtree(spacers[others]).query(per_label[i], predicate='intersects')
        a_parts.append(q + start[i])
        b_parts.append(others[t])
    a = np.concatenate(a_parts) if a_parts else np.empty(0, dtype=int)
    b = np.concatenate(b_parts) if b_parts else np.empty(0, dtype=int)
    order = np.argsort(a, kind='stable')
    a, b = a[order], b[order]
    indptr = np.searchsorted(a, np.arange(len(keys) + 1))
    nbrs = lambda g: b[indptr[g]:indptr[g + 1]]
    alive = np.ones(len(keys), dtype=bool)
    fixed: Dict[int, int] = {}

    def best(i: int, banned: np.ndarray = None) -> Optional[int]:
        idx = np.arange(start[i], start[i + 1])
        mask = alive[idx]
        if banned is not None:
            mask &= ~np.isin(idx, banned)
        if not mask.any():
            return None
        return int(idx[mask][np.argmin(keys[idx][mask])])

    while True:
        open_ = [i for i in range(len(lids)) if i not in fixed]
        picks = {i: best(i) for i in open_}
        picks = {i: g for i, g in picks.items() if g is not None}
        if not picks:
            break
        pick_set = set(picks.values())
        partners = {i: [owner[h] for h in nbrs(g) if h in pick_set] for i, g in picks.items()}
        clashing = [i for i in picks if partners[i]]
        if not clashing:
            fixed.update(picks)
            break
        loss = {}
        for i in clashing:
            banned = np.concatenate([nbrs(picks[j]) for j in partners[i]])
            alt = best(i, banned)
            loss[i] = np.inf if alt is None else keys[alt] - keys[picks[i]]
        i = max(clashing, key=lambda j: (loss[j], -keys[picks[j]]))
        fixed[i] = picks[i]
        alive[np.arange(start[i], start[i + 1])] = False
        alive[nbrs(picks[i])] = False
    return {lids[i]: int(top[lids[i]][g - start[i]]) for i, g in fixed.items()}

def _with_binder(
        layout: Layout,
        lid: int,
        candidate: Candidate,
        spots: Tuple[np.ndarray, np.ndarray],
        result: Dict[int, Candidate],
        placed: Dict[int, Candidate],
        context: _Context,
        cfg: Config
) -> Candidate:
    '''
    Binder for a placed overflow label: straight if conflict-free (the dot may
    slide along its ring to points at most `ATTACH_SLACK_MM` farther), else a
    curve (soft rules, no binder crossings). If no binder is acceptable, the
    next best positions are tried (Q9.5); if none works, the first position keeps
    the binder with the fewest conflicts.

    Parameters
    ----------
    layout : Layout
        fixed context
    lid : int
        label id
    candidate : Candidate
        label at its chosen position
    spots : Tuple[np.ndarray, np.ndarray]
        positions and keys of the label
    result : Dict[int, Candidate]
        overflow labels; those with a binder are placed
    placed : Dict[int, Candidate]
        chosen positions of all overflow labels
    context : _Context
        free-space context
    cfg : Config
        configuration

    Returns
    -------
    candidate : Candidate
        label with position and binder
    '''
    r = route(layout, lid, candidate, result, cfg, soft=True, attachment_slack=ATTACH_SLACK_MM / MM_PER_UNIT, allow_binder_crossings=False)
    if r is not None:
        return candidate.moved(candidate.position, r[0])
    P, keys = spots
    others = [c for o, c in placed.items() if o != lid]
    order = np.argsort(keys, kind='stable')
    ok = context.feasible(candidate, P[order], others)
    for p in P[order][ok][:MAX_POSITION_TRIES]:
        c = candidate.moved(p)
        r = route(layout, lid, c, result, cfg, soft=True, attachment_slack=ATTACH_SLACK_MM / MM_PER_UNIT, allow_binder_crossings=False)
        if r is not None:
            placed[lid] = c
            return c.moved(c.position, r[0])
    r = route(layout, lid, candidate, result, cfg, soft=True, attachment_slack=ATTACH_SLACK_MM / MM_PER_UNIT, any_binder=True, allow_binder_crossings=False)
    return candidate.moved(candidate.position, r[0])

@lru_cache(maxsize=None)
def _node_keep(shape: LabelShape, keep: float) -> BaseGeometry:
    '''
    Letters grown by the node marker radius + node gap (cached per shape): no node
    centre may lie inside.

    Parameters
    ----------
    shape : LabelShape
        label shape
    keep : float
        distance

    Returns
    -------
    grown : BaseGeometry
        prepared grown letters
    '''
    grown = shape.ink.buffer(keep)
    shapely.prepare(grown)
    return grown
