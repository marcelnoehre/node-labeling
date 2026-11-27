import numpy as np
import networkx as nx
import shapely

from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Set, Tuple

from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree
from shapely.validation import make_valid

from node_labeling.utils.config import Config
from node_labeling.utils.constants import BINDER_DOT_SIZE, DOT_CLEARANCE_MM, LINE_WIDTH, MM_PER_UNIT, NODE_RADIUS, NODE_SIZE
from node_labeling.v2.filter.shape_rules import Obstacles
from node_labeling.v2.models.binder import Binder
from node_labeling.v2.models.candidate import Candidate

ATTACH_SAMPLES: int = 32
OWN_EPS: float = 1e-6
BEND_STEP: float = 7.5
MIN_EDGE_ANGLE: float = 15.0
CONTROL_FRACTION: float = 1.0 / 3.0
MAX_ROUTE_TRIES: int = 30
DOT_RADIUS: float = np.sqrt(BINDER_DOT_SIZE) / 2.0 * 25.4 / 72.0 / MM_PER_UNIT
MARKER_RADIUS: float = (np.sqrt(NODE_SIZE) / 2.0 + LINE_WIDTH / 2.0) * 25.4 / 72.0 / MM_PER_UNIT
ROUTE_ALTERNATIVES: int = 8

@dataclass
class Layout:
    '''
    Fixed context of the overflow stages: the diagram and the labels placed
    directly at their nodes.

    Attributes
    ----------
    G : nx.Graph
        planarized graph containing normalized positions
    obstacles : Obstacles
        edges, nodes, hull
    direct : List[Candidate]
        labels placed at their nodes
    ink_tree : STRtree
        index over the inks of `direct`
    pad_tree : STRtree
        index over the pads of `direct`
    body_tree : STRtree
        index over the bodies of `direct` (letters with closed gaps + edge clearance)
    face_tree : STRtree
        index over the bounded faces
    dual : Dict[int, Set[int]]
        faces sharing an edge (-1 = outer face)
    faces_of : Dict[Hashable, Set[int]]
        faces around every node
    '''
    G: nx.Graph
    obstacles: Obstacles
    direct: List[Candidate]
    ink_tree: STRtree
    pad_tree: STRtree
    body_tree: STRtree
    face_tree: STRtree
    dual: Dict[int, Set[int]]
    faces_of: Dict[Hashable, Set[int]]
    _crossings: Dict[Tuple[Hashable, int], int] = field(default_factory=dict)

    @classmethod
    def build(cls, G: nx.Graph, obstacles: Obstacles, direct: List[Candidate],
              bounded_faces: List[List[Hashable]], alpha_shape: List[Hashable]) -> 'Layout':
        '''
        Parameters
        ----------
        G : nx.Graph
            planarized graph
        obstacles : Obstacles
            fixed obstacles
        direct : List[Candidate]
            labels placed at their nodes
        bounded_faces : List[List[Hashable]]
            boundary walk of every bounded face
        alpha_shape : List[Hashable]
            boundary walk of the alpha shape

        Returns
        -------
        layout : Layout
            context with spatial indexes and the dual graph of the faces
        '''
        inks = [c.ink for c in direct]
        pads = [c.pad for c in direct]
        bodies = [c.body for c in direct]
        shapely.prepare(inks + pads + bodies)
        polys = []
        for face in bounded_faces:
            poly = Polygon([G.nodes[n]['pos'] for n in face])
            polys.append(poly if poly.is_valid else make_valid(poly))
        edge_faces: Dict[frozenset, List[int]] = {}
        faces_of: Dict[Hashable, Set[int]] = {n: set() for n in G.nodes}
        for f, face in enumerate(bounded_faces):
            for i, n in enumerate(face):
                faces_of[n].add(f)
                edge_faces.setdefault(frozenset((n, face[(i + 1) % len(face)])), []).append(f)
        for n in alpha_shape:
            faces_of[n].add(-1)
        dual: Dict[int, Set[int]] = {f: set() for f in range(-1, len(bounded_faces))}
        for fs in edge_faces.values():
            fs = fs if len(fs) > 1 else fs + [-1]
            for a in fs:
                dual[a].update(b for b in fs if b != a)
        return cls(G, obstacles, direct, STRtree(inks), STRtree(pads), STRtree(bodies), STRtree(polys), dual, faces_of)

    def min_crossings(self, node_id: Hashable, point: Tuple[float, float]) -> int:
        '''
        Fewest edges a line from a node to a point has to cross: the distance in
        the dual graph from the faces around the node to the face of the point.

        Parameters
        ----------
        node_id : Hashable
            node
        point : Tuple[float, float]
            end point

        Returns
        -------
        crossings : int
            unavoidable edge crossings
        '''
        hit = self.face_tree.query(Point(point), predicate='within')
        target = int(hit[0]) if len(hit) else -1
        key = (node_id, target)
        if key not in self._crossings:
            start = self.faces_of.get(node_id, set())
            depth, seen, frontier = 0, set(start), set(start)
            while frontier and target not in frontier:
                frontier = {b for a in frontier for b in self.dual[a]} - seen
                seen |= frontier
                depth += 1
            self._crossings[key] = depth if target in seen else 0
        return self._crossings[key]

    def node_pos(self, node_id: Hashable) -> Tuple[float, float]:
        '''
        Parameters
        ----------
        node_id : Hashable
            node

        Returns
        -------
        pos : Tuple[float, float]
            node position
        '''
        x, y = self.G.nodes[node_id]['pos']
        return float(x), float(y)

def attach(candidate: Candidate) -> Binder:
    '''
    Shortest binder of an overflow label: from its node to the nearest point of
    its dot ring (ink grown by dot radius + gap). It never crosses its own label.

    Parameters
    ----------
    candidate : Candidate
        overflow label at its position

    Returns
    -------
    binder : Binder
        straight binder
    '''
    node = Point(candidate.node_pos)
    ring = candidate.dot_ring.exterior
    a = ring.interpolate(ring.project(node))
    return Binder(start=candidate.node_pos, end=(a.x, a.y))

def attach_many(candidates: List[Candidate]) -> List[Binder]:
    '''
    `attach` for many positions of one label at once.

    Parameters
    ----------
    candidates : List[Candidate]
        the same label at different positions

    Returns
    -------
    binders : List[Binder]
        shortest binder per position
    '''
    if not candidates:
        return []
    shape = candidates[0].shape
    pos = np.array([c.position for c in candidates])
    q = np.asarray(candidates[0].node_pos) - pos
    lines = shapely.shortest_line(shape.dot_ring.exterior, shapely.points(q))
    ends = shapely.get_coordinates(lines).reshape(-1, 2, 2)[:, 0] + pos
    return [Binder(start=c.node_pos, end=(float(x), float(y))) for c, (x, y) in zip(candidates, ends)]

def alternatives(candidate: Candidate, samples: int = ATTACH_SAMPLES) -> List[Binder]:
    '''
    Binders to other points of the dot ring that do not pass through the own
    label, shortest first (the first one is `attach`).

    Parameters
    ----------
    candidate : Candidate
        overflow label at its position
    samples : int
        number of ring points tried

    Returns
    -------
    binders : List[Binder]
        straight binders sorted by length
    '''
    ext = candidate.shape.dot_ring.exterior
    local = shapely.get_coordinates(shapely.line_interpolate_point(ext, np.linspace(0, 1, samples, endpoint=False), normalized=True))
    pts = local + np.asarray(candidate.position)
    node = np.asarray(candidate.node_pos)
    d = pts - node
    n = np.hypot(*d.T)
    keep = n > OWN_EPS
    pts, d, n = pts[keep], d[keep], n[keep]
    stubs = shapely.linestrings(np.stack([np.repeat(node[None], len(pts), 0), pts - d / n[:, None] * OWN_EPS], axis=1))
    ok = ~shapely.intersects(stubs, candidate.dot_ring)
    order = np.argsort(n, kind='stable')
    return [attach(candidate)] + [Binder(start=candidate.node_pos, end=(float(x), float(y))) for x, y in pts[order][ok[order]]]

@dataclass(frozen=True)
class Conflicts:
    '''
    Conflicts of one binder (Q8.2).

    Attributes
    ----------
    hard : int
        hits of other labels (strict: pads, soft: bodies = letters with closed gaps + edge clearance) and placed
        overflow labels, nodes closer than
        `NODE_RADIUS`, attachment closer than `NODE_RADIUS` to a forbidden edge, a binder dot that comes closer
        than `DOT_CLEARANCE_MM` to anything (edges, node markers, letters, binders, dots); curves also:
        passing through the own label, loops, leaving the node along one of its edges
    binder_cross : int
        crossings with binders of placed overflow labels
    edge_cross : int
        crossings with edges
    edge_min : int
        edge crossings no binder from the node to the attachment can avoid
    '''
    hard: int
    binder_cross: int
    edge_cross: int
    edge_min: int = 0

    @property
    def crossings(self) -> int:
        '''
        Returns
        -------
        crossings : int
            binder crossings and avoidable edge crossings (the conflicts a curve could remove)
        '''
        return self.binder_cross + max(0, self.edge_cross - self.edge_min)

    @property
    def free(self) -> bool:
        '''
        Returns
        -------
        free : bool
            whether the binder has no conflict at all
        '''
        return self.hard == 0 and self.crossings == 0

    def acceptable(self, soft: bool, max_edge_crossings: int = -1) -> bool:
        '''
        Parameters
        ----------
        soft : bool
            soft mode allows binder crossings
        max_edge_crossings : int
            maximal number of edge crossings, -1 for no limit

        Returns
        -------
        acceptable : bool
            whether a stage may use the binder
        '''
        if self.hard or (not soft and self.binder_cross):
            return False
        return max_edge_crossings < 0 or self.edge_cross <= max_edge_crossings

def conflicts(
        layout: Layout,
        lid: int,
        candidate: Candidate,
        binder: Binder,
        overflow: Dict[int, Candidate],
        soft: bool = False,
        edges: Optional[List[Tuple[Hashable, Hashable]]] = None
) -> Conflicts:
    '''
    Count the conflicts of a binder (v1's `binding_line_valid` rules on shapes, as counts).

    Parameters
    ----------
    layout : Layout
        fixed context
    lid : int
        label id of the binder's label
    candidate : Candidate
        the label at its position
    binder : Binder
        binder to check
    overflow : Dict[int, Candidate]
        overflow labels; those with a binder count as placed
    soft : bool
        soft mode: other labels count by their bodies (letters with closed gaps + edge clearance), not their
        pads (overflow labels: exp shapes)
    edges : Optional[List[Tuple[Hashable, Hashable]]]
        edges the attachment keeps away from

    Returns
    -------
    conflicts : Conflicts
        hard conflicts and crossings
    '''
    line = binder.line
    hard = binder_cross = 0
    for o_lid, other in overflow.items():
        if o_lid == lid or other.binder is None:
            continue
        hard += line.intersects(other.body if soft else other.exp)
        binder_cross += line.crosses(other.binder.line)

    tree = layout.body_tree if soft else layout.pad_tree
    hard += len(tree.query(line, predicate='intersects'))

    obst = layout.obstacles
    others = np.array([nid != candidate.node_id for nid in obst.node_ids], dtype=bool)
    if others.any():
        hard += int((shapely.distance(line, shapely.points(obst.node_xy[others])) < NODE_RADIUS).sum())

    end = Point(binder.end)
    for a, b in edges or []:
        hard += end.distance(LineString([layout.node_pos(a), layout.node_pos(b)])) < NODE_RADIUS

    hard += _dot_fault(layout, lid, candidate, binder, overflow)
    if not binder.straight:
        hard += _curve_faults(layout, candidate, binder)
    edge_cross = len(obst.edge_tree.query(line, predicate='crosses'))
    edge_min = layout.min_crossings(candidate.node_id, binder.end)
    return Conflicts(int(hard), int(binder_cross), int(edge_cross), int(edge_min))

def binder_valid(
        layout: Layout,
        lid: int,
        candidate: Candidate,
        binder: Binder,
        overflow: Dict[int, Candidate],
        soft: bool = False,
        edges: Optional[List[Tuple[Hashable, Hashable]]] = None,
        max_edge_crossings: int = -1
) -> bool:
    '''
    Check a binder (v1's `binding_line_valid` on shapes): no hard conflict, no
    binder crossing unless soft, at most `max_edge_crossings` edge crossings.

    Parameters
    ----------
    layout : Layout
        fixed context
    lid : int
        label id of the binder's label
    candidate : Candidate
        the label at its position
    binder : Binder
        binder to check
    overflow : Dict[int, Candidate]
        overflow labels (those with a binder are placed)
    soft : bool
        less restrictive mode
    edges : Optional[List[Tuple[Hashable, Hashable]]]
        edges the attachment keeps away from
    max_edge_crossings : int
        maximal number of edge crossings, -1 for no limit

    Returns
    -------
    valid : bool
        whether the binder is valid
    '''
    return conflicts(layout, lid, candidate, binder, overflow, soft, edges).acceptable(soft, max_edge_crossings)

def _dot_fault(layout: Layout, lid: int, candidate: Candidate, binder: Binder, overflow: Dict[int, Candidate]) -> int:
    '''
    Whether the binder dot comes closer than `DOT_CLEARANCE_MM` to an edge, a node
    marker, the letters of another label, another binder or its dot.

    Parameters
    ----------
    layout : Layout
        fixed context
    lid : int
        label id of the binder's label
    candidate : Candidate
        the label at its position
    binder : Binder
        binder
    overflow : Dict[int, Candidate]
        overflow labels (those with a binder are placed)

    Returns
    -------
    fault : int
        1 if the dot touches something, else 0
    '''
    clear = DOT_CLEARANCE_MM / MM_PER_UNIT
    end = Point(binder.end)
    disc = end.buffer(DOT_RADIUS + clear)
    obst = layout.obstacles
    if len(obst.edge_tree.query(disc, predicate='intersects')):
        return 1
    if (np.hypot(*(obst.node_xy - np.asarray(binder.end)).T) < MARKER_RADIUS + DOT_RADIUS + clear).any():
        return 1
    if len(layout.ink_tree.query(disc, predicate='intersects')):
        return 1
    for o_lid, other in overflow.items():
        if o_lid == lid or other.binder is None:
            continue
        if disc.intersects(other.ink) or disc.intersects(other.binder.line):
            return 1
        if end.distance(Point(other.binder.end)) < 2 * DOT_RADIUS + clear:
            return 1
    return 0

def _curve_faults(layout: Layout, candidate: Candidate, binder: Binder) -> int:
    '''
    Curve-only hard conflicts: a loop, passing through the own label (the part
    inside the own pad must be one piece ending at the dot), leaving the node at
    less than `MIN_EDGE_ANGLE` to one of its edges.

    Parameters
    ----------
    layout : Layout
        fixed context
    candidate : Candidate
        the label at its position
    binder : Binder
        curved binder

    Returns
    -------
    faults : int
        number of faults
    '''
    line = binder.line
    faults = int(not line.is_simple)
    inside = line.intersection(candidate.pad)
    faults += inside.geom_type != 'LineString' or inside.is_empty
    tangent = binder.start_tangent
    node = np.asarray(binder.start)
    G = layout.G
    for nb in G.neighbors(candidate.node_id):
        d = np.asarray(G.nodes[nb]['pos']) - node
        n = np.linalg.norm(d)
        if n > 0 and np.degrees(np.arccos(np.clip(np.dot(tangent, d / n), -1.0, 1.0))) < MIN_EDGE_ANGLE:
            faults += 1
            break
    return faults

def curve(candidate: Candidate, alpha: float, beta: float) -> Binder:
    '''
    Cubic binder leaving the node at `alpha` and arriving at the label at `beta`
    (degrees, relative to the chord; same sign = one bend, opposite signs = S).
    The control points sit `CONTROL_FRACTION` of the chord along the end
    directions. The attachment is the dot ring point nearest to the second
    control point, so the curve arrives from outside (one re-pick).

    Parameters
    ----------
    candidate : Candidate
        overflow label at its position
    alpha : float
        leaving angle at the node
    beta : float
        arriving angle at the label

    Returns
    -------
    binder : Binder
        curved binder
    '''
    p0 = np.asarray(candidate.node_pos, dtype=float)
    ring = candidate.dot_ring.exterior

    def controls(p3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        chord = p3 - p0
        length = np.linalg.norm(chord)
        u = chord / length if length > 0 else np.array([1.0, 0.0])
        return p0 + CONTROL_FRACTION * length * _rotate(u, alpha), p3 - CONTROL_FRACTION * length * _rotate(u, -beta)

    p3 = np.asarray(attach(candidate).end)
    _, p2 = controls(p3)
    a = ring.interpolate(ring.project(Point(p2)))
    p3 = np.array([a.x, a.y])
    p1, p2 = controls(p3)
    return Binder(start=candidate.node_pos, end=(float(p3[0]), float(p3[1])),
                  controls=((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))))

def curve_angles(binder: Binder) -> Tuple[float, float]:
    '''
    Parameters
    ----------
    binder : Binder
        binder

    Returns
    -------
    alpha, beta : Tuple[float, float]
        leaving and arriving angle in degrees relative to the chord (0, 0 for straight binders)
    '''
    if binder.straight:
        return 0.0, 0.0
    p0, p3 = np.asarray(binder.start), np.asarray(binder.end)
    p1, p2 = (np.asarray(c) for c in binder.controls)
    chord = p3 - p0
    return _signed_angle(chord, p1 - p0), _signed_angle(p3 - p2, chord)

def binder_cost(binder: Binder, cfg: Config) -> float:
    '''
    Cost of a binder (Q8.3): length · `w_binder` + bend (radians) · `w_binder_bend`,
    + `w_binder_s` for S-curves.

    Parameters
    ----------
    binder : Binder
        binder
    cfg : Config
        configuration

    Returns
    -------
    cost : float
        binder cost
    '''
    cost = binder.length * cfg.w_binder + np.radians(binder.bend) * cfg.w_binder_bend
    return cost + (cfg.w_binder_s if binder.kind == 's' else 0.0)

def route(
        layout: Layout,
        lid: int,
        candidate: Candidate,
        overflow: Dict[int, Candidate],
        cfg: Config,
        soft: bool = False,
        edges: Optional[List[Tuple[Hashable, Hashable]]] = None,
        max_edge_crossings: int = -1,
        other_attachments: bool = True,
        any_binder: bool = False,
        allow_binder_crossings: bool = True,
        attachment_slack: float = np.inf
) -> Optional[Tuple[Binder, Conflicts]]:
    '''
    Best binder of an overflow label at its position (T8.2).

    The shortest straight binder is used if it has no conflict. Otherwise the
    options are tried cheapest first (by a lower bound of their cost): straight
    binders to `ROUTE_ALTERNATIVES` other dot ring points, one-bend curves (bend in
    steps of `BEND_STEP` up to `cfg.max_binder_bend`, both sides; symmetric and
    two asymmetric ones per step), symmetric S-curves. The best one
    has the fewest hard conflicts, then the fewest crossings, then the lowest cost;
    a straight binder wins ties. The search stops once a conflict-free option is
    found and no cheaper one is left, or after `MAX_ROUTE_TRIES` options.

    Parameters
    ----------
    layout : Layout
        fixed context
    lid : int
        label id
    candidate : Candidate
        overflow label at its position
    overflow : Dict[int, Candidate]
        overflow labels (those with a binder are placed)
    cfg : Config
        configuration (curved binders, bend cap, costs)
    soft, edges, max_edge_crossings
        stage rules (see `Conflicts.acceptable`)
    other_attachments : bool
        whether straight binders to other dot ring points are options
    any_binder : bool
        return the best option even if none is acceptable
    allow_binder_crossings : bool
        whether crossings with other binders are acceptable (soft mode allows them otherwise)
    attachment_slack : float
        other attachment points only if their straight binder is at most this much longer than the shortest

    Returns
    -------
    route : Optional[Tuple[Binder, Conflicts]]
        best acceptable binder with its conflicts, None if no option is acceptable (unless `any_binder`)
    '''
    straight = attach(candidate)
    found = [(straight, conflicts(layout, lid, candidate, straight, overflow, soft, edges))]
    if found[0][1].free:
        return found[0]

    near = [b for b in alternatives(candidate)[1:] if b.length <= straight.length + attachment_slack] if other_attachments else []
    options = [(b.length * cfg.w_binder, 0, i, lambda b=b: b) for i, b in enumerate(near[:ROUTE_ALTERNATIVES])]
    if cfg.curved_binders:
        base = straight.length * cfg.w_binder
        levels = np.arange(BEND_STEP, cfg.max_binder_bend + 1e-9, BEND_STEP)
        for b in levels:
            bound = base + np.radians(b) * cfg.w_binder_bend
            for side in (1, -1):
                for a1, b1 in ((b, b), (b, b / 2), (b / 2, b)):
                    options.append((bound, 1, len(options), lambda a1=a1, b1=b1, side=side: curve(candidate, side * a1, side * b1)))
                options.append((bound + cfg.w_binder_s, 2, len(options), lambda b=b, side=side: curve(candidate, side * b, -side * b)))
    options.sort(key=lambda o: o[:3])

    def key(item: Tuple[Binder, Conflicts]):
        b, c = item
        return (c.hard, c.crossings, binder_cost(b, cfg), not b.straight)

    best_free = None
    for bound, _, _, make in options[:MAX_ROUTE_TRIES]:
        if best_free is not None and bound >= binder_cost(best_free, cfg):
            break
        b = make()
        c = conflicts(layout, lid, candidate, b, overflow, soft, edges)
        found.append((b, c))
        if c.free and (best_free is None or binder_cost(b, cfg) < binder_cost(best_free, cfg)):
            best_free = b
    usable = [item for item in found if item[1].acceptable(soft, max_edge_crossings)
              and (allow_binder_crossings or not item[1].binder_cross)]
    if not usable and any_binder:
        usable = found
    return min(usable, key=key) if usable else None

def straighten(
        layout: Layout,
        overflow: Dict[int, Candidate],
        strict: List[int],
        cfg: Config,
        edges_of=None
) -> Dict[int, Candidate]:
    '''
    Final pass (T8.4): a curved binder becomes straight if the straight binder has
    no more conflicts (hard, then crossings) in the final layout; otherwise the
    smallest bend that keeps the conflicts is used (S-curves become one bend
    where possible).

    Parameters
    ----------
    layout : Layout
        fixed context
    overflow : Dict[int, Candidate]
        overflow labels
    strict : List[int]
        labels placed with the strict rules (inside faces); the others use the soft rules
    cfg : Config
        configuration
    edges_of : Optional[Callable[[int], List]]
        forbidden attachment edges per strict label

    Returns
    -------
    overflow : Dict[int, Candidate]
        overflow labels with straightened binders
    '''
    for lid, c in overflow.items():
        if c.binder is None or c.binder.straight:
            continue
        soft = lid not in strict
        edges = edges_of(lid) if (edges_of and not soft) else None
        current = conflicts(layout, lid, c, c.binder, overflow, soft, edges)
        alpha, beta = curve_angles(c.binder)
        trials = [attach(c)]
        if c.binder.kind == 's':
            trials.append(curve(c, alpha, 0.0) if abs(alpha) >= abs(beta) else curve(c, 0.0, beta))
        trials += [curve(c, alpha * f, beta * f) for f in (0.25, 0.5, 0.75)]
        for b in trials:
            new = conflicts(layout, lid, c, b, overflow, soft, edges)
            if (new.hard, new.crossings) <= (current.hard, current.crossings) and new.acceptable(soft):
                overflow[lid] = c.moved(c.position, b)
                break
    return overflow

def _rotate(u: np.ndarray, degrees: float) -> np.ndarray:
    t = np.radians(degrees)
    return np.array([u[0] * np.cos(t) - u[1] * np.sin(t), u[0] * np.sin(t) + u[1] * np.cos(t)])

def _signed_angle(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(u[0] * v[1] - u[1] * v[0], np.dot(u, v))))
