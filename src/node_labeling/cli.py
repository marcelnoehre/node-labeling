import argparse
import copy
import time

from pathlib import Path
from typing import Dict, Optional

from node_labeling.utils.config import Config
from node_labeling.utils.visualize import plot_graph
from node_labeling.utils.normalize import *
from node_labeling.utils.graph_lattice import GraphLattice
from node_labeling.models.label_candidate import LabelCandidate
from node_labeling.topology.planarize import *
from node_labeling.topology.faces import *
from node_labeling.label.generate_candidates import *
from node_labeling.filter.filter import *
from node_labeling.label.generate_overflow import *
from node_labeling.overflow.bounded import *
from node_labeling.overflow.unbounded import *
from node_labeling.forces.forces import *

def label(graphml: str, cfg: Optional[Config] = None, export: bool = False) -> Optional[Dict[int, LabelCandidate]]:
    '''
    Run the node labeling pipeline end to end.

    Parameters
    ----------
    graphml : str
        Path to a .graphml file. Each node needs `x`/`y` position data, and
        optionally `general` data as a plain string and `extent`/`intent`
        label data (`;`-separated names).
    cfg : Config, optional
        Configuration overrides. Defaults to `Config()`.
    export : bool
        If True, write the final layout to a PDF in `figs/` and return None.
        If False, skip the export and return the computed label placements instead.

    Returns
    -------
    labels : Optional[Dict[int, LabelCandidate]]
        The final placed label candidate for each label id, or None if export is True.
    '''
    init_time = time.perf_counter()
    cfg = cfg or Config()

    lattice = GraphLattice(graphml)
    nodes = lattice.nodes
    relations = list(lattice.cover_relations())
    positions = lattice.positions

    # planarize graph
    start_time = time.perf_counter()
    intersections = find_intersections(relations, positions)
    G = build_planar_graph(relations, positions, intersections)
    planarize_duration = (time.perf_counter() - start_time) * 1000
    if cfg.runtime:
        print(f"\033[1;36m[RUNTIME]\033[0m Planarize runtime: \033[1;32m{planarize_duration:.2f} ms\033[0m")

    # normalize positions
    scale = normalize_positions(G)
    intersection_points = normalize_intersections(G, intersections)
    if cfg.plot:
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/input.pdf'
        )
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/intersections.pdf',
            intersections=intersection_points,
            show_intersections=True
        )

    # faces
    start_time = time.perf_counter()
    alpha_shape, bounded_faces, areas, centroids = extract_faces(G, scale)
    topology_duration = (time.perf_counter() - start_time) * 1000
    if cfg.runtime:
        print(f"\033[1;36m[RUNTIME]\033[0m Compute topology: \033[1;32m{topology_duration:.2f} ms\033[0m")
    if cfg.plot:
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/faces.pdf',
            intersections=intersection_points,
            show_intersections=True,
            alpha_shape=alpha_shape,
            show_alpha_shape=True,
            bounded_faces=bounded_faces,
            areas=areas,
            centroids=centroids,
            show_face_areas=True
        )

    # initial set of label candidates
    start_time = time.perf_counter()
    label_candidates = generate_label_candidates(G, lattice, cfg)
    rendering_duration = (time.perf_counter() - start_time) * 1000
    initial_candidates = copy.deepcopy(label_candidates)
    if cfg.runtime:
        print(f"\033[1;36m[RUNTIME]\033[0m Rendering runtime: \033[1;32m{rendering_duration:.2f} ms\033[0m")
    if cfg.plot:
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/all_label_candidates.pdf',
            label_candidates=label_candidates,
            colored_label_candidates=True,
            show_legend=True
        )

    # filter outer nodes
    start_time = time.perf_counter()
    label_candidates = restrict_outer_node_candidates(G, label_candidates, alpha_shape)
    filter_out_duration = (time.perf_counter() - start_time) * 1000
    if cfg.runtime:
        print(f"\033[1;36m[RUNTIME]\033[0m Filter out runtime: \033[1;32m{filter_out_duration:.2f} ms\033[0m")
    if cfg.plot:
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/filtered_outer.pdf',
            label_candidates=label_candidates,
            colored_label_candidates=True
        )

    # filter unclear node assignment
    start_time = time.perf_counter()
    label_candidates = filter_candidates_by_nodes(G, label_candidates, nodes)
    filter_node_duration = (time.perf_counter() - start_time) * 1000
    if cfg.runtime:
        print(f"\033[1;36m[RUNTIME]\033[0m Filter node runtime: \033[1;32m{filter_node_duration:.2f} ms\033[0m")
    if cfg.plot:
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/filter_node.pdf',
            label_candidates=label_candidates,
            colored_label_candidates=True
        )

    # filter overlapping edges
    start_time = time.perf_counter()
    label_candidates = filter_candidates_by_edges(G, label_candidates, bounded_faces)
    filter_edge_duration = (time.perf_counter() - start_time) * 1000
    if cfg.runtime:
        print(f"\033[1;36m[RUNTIME]\033[0m Filter edge runtime: \033[1;32m{filter_edge_duration:.2f} ms\033[0m")
    if cfg.plot:
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/filter_edges.pdf',
            label_candidates=label_candidates,
            colored_label_candidates=True
        )

    # filter by neighbor direction
    start_time = time.perf_counter()
    label_candidates = filter_candidates_by_neighbor_direction(G, label_candidates, lattice)
    filter_neighbor_duration = (time.perf_counter() - start_time) * 1000
    if cfg.runtime:
        print(f"\033[1;36m[RUNTIME]\033[0m Filter neighbor runtime: \033[1;32m{filter_neighbor_duration:.2f} ms\033[0m")
    if cfg.plot:
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/filtered_neighbor.pdf',
            label_candidates=label_candidates,
            colored_label_candidates=True
        )

    # filter by hybrid algorithm
    start_time = time.perf_counter()
    label_candidates = filter_hyrid(label_candidates)
    filter_hybrid_duration = (time.perf_counter() - start_time) * 1000
    if cfg.runtime:
        print(f"\033[1;36m[RUNTIME]\033[0m Filter hybrid runtime: \033[1;32m{filter_hybrid_duration:.2f} ms\033[0m")
    if cfg.plot:
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/filtered_hybrid.pdf',
            label_candidates=label_candidates,
            colored_label_candidates=True
        )

    overflow_candidates = generate_overflow_candidates(G, label_candidates, initial_candidates)
    if cfg.plot:
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/all_overflow_candidates.pdf',
            label_candidates=label_candidates,
            colored_label_candidates=True,
            overflow_candidates=overflow_candidates
        )

    start_time = time.perf_counter()
    overflow_candidates = bounded_overflow_labels(G, label_candidates, overflow_candidates, bounded_faces, centroids, alpha_shape)
    bounded_overflow_duration = (time.perf_counter() - start_time) * 1000
    if cfg.runtime:
        print(f"\033[1;36m[RUNTIME]\033[0m Bounded overflow runtime: \033[1;32m{bounded_overflow_duration:.2f} ms\033[0m")
    if cfg.plot:
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/bounded_overflow_candidates.pdf',
            label_candidates=label_candidates,
            colored_label_candidates=True,
            overflow_candidates=overflow_candidates
        )

    unbounded: List[int] = [lid for lid, ol in overflow_candidates.items() if ol.anchor.anchor_type == AnchorType.O]
    grid_candidates, overflow_candidates, grid_duration, hungarian_duration = unbounded_overflow_labels(G, label_candidates, overflow_candidates, alpha_shape, cfg)
    if cfg.runtime:
        print(f"\033[1;36m[RUNTIME]\033[0m Grid creation runtime: \033[1;32m{grid_duration:.2f} ms\033[0m")
        print(f"\033[1;36m[RUNTIME]\033[0m Hungarian solver runtime: \033[1;32m{hungarian_duration:.2f} ms\033[0m")
    if cfg.plot:
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/unbounded_grid_candidates.pdf',
            label_candidates=label_candidates,
            colored_label_candidates=True,
            overflow_candidates=overflow_candidates,
            grid_candidates=grid_candidates
        )
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/unbounded_overflow_candidates.pdf',
            label_candidates=label_candidates,
            colored_label_candidates=True,
            overflow_candidates=overflow_candidates
        )

    start_time = time.perf_counter()
    overflow_candidates = optimize_overflow_labels(G, label_candidates, overflow_candidates, unbounded, alpha_shape, cfg)
    force_duration = (time.perf_counter() - start_time) * 1000
    if cfg.runtime:
        print(f"\033[1;36m[RUNTIME]\033[0m Force refinement runtime: \033[1;32m{force_duration:.2f} ms\033[0m")
    if cfg.plot:
        plot_graph(
            G,
            nodes,
            relations,
            output_path='figs/force_refined.pdf',
            label_candidates=label_candidates,
            colored_label_candidates=True,
            overflow_candidates=overflow_candidates
        )

    total_duration = (time.perf_counter() - init_time) * 1000
    if cfg.runtime:
        print(f"\033[1;36m[RUNTIME]\033[0m Total runtime: \033[1;32m{total_duration:.2f} ms\033[0m")

    if export:
        Path('figs').mkdir(exist_ok=True)
        plot_graph(
            G,
            nodes,
            relations,
            output_path=f'figs/{Path(graphml).stem}.pdf',
            label_candidates=label_candidates,
            overflow_candidates=overflow_candidates
        )
        return None

    labels: Dict[int, LabelCandidate] = {
        **{lid: candidates[0] for lid, candidates in label_candidates.items() if candidates},
        **overflow_candidates
    }
    return labels


def main():
    parser = argparse.ArgumentParser(description='Node Labeling: two-phase labeling of line diagrams of ordered sets')
    parser.add_argument(
        '--export', action='store_true',
        help='Write the final layout to a PDF in figs/ instead of printing the label placements'
    )
    args = parser.parse_args()

    graphml = input('Path to .graphml file: ')
    labels = label(graphml, export=args.export)
    if labels is not None:
        print(labels)


if __name__ == '__main__':
    main()
