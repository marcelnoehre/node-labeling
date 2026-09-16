from typing import Dict, List, Tuple
from node_labeling.models.anchor import AnchorType
from node_labeling.models.label_candidate import LabelCandidate
from node_labeling.utils.geometry import pad_overlap
from node_labeling.filter.rules import *

def connected_components(
        label_candidates: Dict[int, List[LabelCandidate]],
        conflicts: Dict[Tuple[int, AnchorType], List[Tuple[int, LabelCandidate]]],
) -> List[Dict[int, List[LabelCandidate]]]:
    '''
    Group features into connected components via their conflict graph.

    Parameters
    ----------
    label_candidates : Dict[int, List[LabelCandidate]],
        active label candidates
    conflicts : Dict[Tuple[int, AnchorType], List[Tuple[int, LabelCandidate]]]
        conflicting label candidates

    Returns
    -------
    components : Dict[int, List[LabelCandidate]]
        connected components
    '''
    lids = list(label_candidates.keys())
    seen: Set[int] = set()
    components = []

    for init in lids:
        if init in seen:
            continue

        comp_lids: Set[int] = set()
        queue = [init]
        while queue:
            lid = queue.pop()
            if lid in comp_lids:
                continue

            comp_lids.add(lid)
            for cand in label_candidates[lid]:
                for nb_lid, _ in conflicts[(lid, cand.anchor.anchor_type)]:
                    if nb_lid not in comp_lids and nb_lid in label_candidates:
                        queue.append(nb_lid)

        seen.update(comp_lids)
        components.append({lid: label_candidates[lid] for lid in comp_lids})

    return components

def maximum_bipartite_matching(
        labels: List[int],
        cliques: List[List[Tuple[int, LabelCandidate]]],
) -> Dict[int, LabelCandidate]:
    '''
    Augmenting-path bipartite matching: feature lids <-> cliques.

    Parameters
    ----------
    labels : List[int]
        labels representing the left side of the bipartite graph
    cliques : List[List[Tuple[int, LabelCandidate]]]
        cliques representing the right side of the bipartite graph

    Returns
    -------
    mapped_candidates : Dict[int, LabelCandidate]
        mapping of label id and candidate

    '''
    ladj: List[List[int]] = [
        [ci for ci, cl in enumerate(cliques) if any(lid == fid for lid, _ in cl)]
        for fid in labels
    ]
    match_l = [-1] * len(labels)
    match_r = [-1] * len(cliques)

    def _dfs(u: int, vis: List[bool]) -> bool:
        '''
        Depth-First Search.

        Parameters
        ----------
        u : int
            current node to be matched.
        vis : List[bool]
            tracking list to prevent re-visiting the same cliques 

        Returns
        -------
        aug_path_found : bool
            wether an augmenting path was found
        '''
        for v in ladj[u]:
            if vis[v]:
                continue

            vis[v] = True
            if match_r[v] == -1 or _dfs(match_r[v], vis):
                match_l[u] = v
                match_r[v] = u
                return True
        
        return False

    for u in range(len(labels)):
        _dfs(u, [False] * len(cliques))

    return {
        labels[u]: next(cand for lid, cand in cliques[match_l[u]] if lid == labels[u])
        for u in range(len(labels))
        if match_l[u] != -1
    }


def kt_reduce(
        components: Dict[int, List[LabelCandidate]],
        conflicts: Dict[Tuple[int, AnchorType], List[Tuple[int, LabelCandidate]]],
) -> List[List[Tuple[int, LabelCandidate]]]:
    '''
    Kakoulis-Tollis: recursively reduce a component into cliques.

    Parameters
    ----------
    components : Dict[int, List[LabelCandidate]]
        connected components
    conflicts : Dict[Tuple[int, AnchorType], List[Tuple[int, LabelCandidate]]]
        conflicting label candidates

    Returns
    -------
    cliques : List[List[Tuple[int, LabelCandidate]]]
        cliques that can potentially be placed together
    '''
    pairs = [(lid, cand) for lid, cands in components.items() for cand in cands]

    if not pairs:
        return []

    if all(
        pad_overlap(ca, cb)
        for i, (_, ca) in enumerate(pairs)
        for _, cb in [pairs[j] for j in range(i + 1, len(pairs))]
    ):
        return [pairs]

    def _degree(lid: int, cand: LabelCandidate):
        '''
        Conflict degree of a label candidate.

        Parameters
        ----------
        lid : int
            candidate being checked
        cand : LabelCandidate
            specific label candidate whose conflicts are being matched

        Returns
        -------
        degree : int
            number of conflicting neighbors currently present in the active components
        '''
        return sum(
            1 for nb_lid, nb_cand in conflicts[(lid, cand.anchor.anchor_type)]
            if nb_lid in components and nb_cand in components[nb_lid]
        )

    best_lid, best_cand = max(
        pairs,
        key=lambda pair: (_degree(*pair), not len(components[pair[0]]) == 1)
    )

    if len(components[best_lid]) == 1:
        best_deg = _degree(best_lid, best_cand)
        alt = next(
            ((lid, cand) for lid, cand in pairs 
                if (lid, cand) != (best_lid, best_cand)
                and not len(components[lid]) == 1
                and _degree(lid, cand) >= best_deg - 1),
            None
        )
        if alt is not None:
            best_lid, best_cand = alt

    remaining: Dict[int, List[LabelCandidate]] = {}
    for lid, cand in pairs:
        if (lid, cand) == (best_lid, best_cand):
            continue

        remaining.setdefault(lid, []).append(cand)

    if not remaining:
        return []

    sub_components = connected_components(remaining, conflicts)
    result = []
    for sub in sub_components:
        result.extend(kt_reduce(sub, conflicts))

    return result
