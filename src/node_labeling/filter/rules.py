import copy
from itertools import combinations
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from node_labeling.models.anchor import AnchorType
from node_labeling.models.label_candidate import LabelCandidate
from node_labeling.utils.geometry import pad_overlap

def build_conflict_map(
        label_candidates: Dict[int, List[LabelCandidate]],
) -> Dict[Tuple[int, LabelCandidate], List[Tuple[int, LabelCandidate]]]:
    '''
    For each candidate find all candidates whose ink box overlaps it.

    Two candidates conflict iff:
      - they belong to different labels
      - their ink boxes strictly intersect

    Parameters
    ----------
    label_candidates : Dict[int, List[LabelCandidate]],
        active label candidates

    Returns
    -------
    conflicts : Dict[Tuple[int, AnchorType], List[Tuple[int, LabelCandidate]]]
        conflicting label candidates
    '''
    conflicts: Dict[Tuple[int, AnchorType], List[Tuple[int, LabelCandidate]]] = defaultdict(list)

    for i, j in list(combinations(list(label_candidates.keys()), 2)):
        for cand_a in label_candidates[i]:
            for cand_b in label_candidates[j]:
                if pad_overlap(cand_a, cand_b):
                    conflicts[(i, cand_a.anchor.anchor_type)].append((j, cand_b))
                    conflicts[(j, cand_b.anchor.anchor_type)].append((i, cand_a))

    return conflicts

def apply_L1(
        label_candidates: Dict[int, List[LabelCandidate]], 
        conflicts: Dict[Tuple[int, AnchorType], List[Tuple[int, LabelCandidate]]]
) -> Tuple[Dict[int, List[LabelCandidate]], bool]:
    '''
    Safe Selection
    From all label candidates that have zero active conflicts with
    any other label, select the one with the lowest readability rank.

    Parameters
    ----------
    label_candidates : Dict[int, List[LabelCandidate]],
        active label candidates
    conflicts : Dict[Tuple[int, AnchorType], List[Tuple[int, LabelCandidate]]]
        conflicting label candidates

    Returns
    -------
    filtered_candidates, changed : Tuple[Dict[int, List[LabelCandidate]], bool]
        reamining label candidates, wether the set of candidates changed
    '''
    filtered_candidates = copy.deepcopy(label_candidates)
    changed = False

    for lid, candidates in label_candidates.items():
        if len(candidates) > 1:
            valid = [
                (lid, cand)
                for cand in candidates
                if not conflicts[(lid, cand.anchor.anchor_type)]
            ]
            if valid:
                valid.sort(key=lambda item: item[1].anchor.anchor_type.rank)
                filtered_candidates[lid] = [valid[0][1]]
                changed = True
    
    return filtered_candidates, changed

def apply_L2(
        label_candidates: Dict[int, List[LabelCandidate]],
        conflicts: Dict[Tuple[int, AnchorType], List[Tuple[int, LabelCandidate]]]
) -> Tuple[Dict[int, List[LabelCandidate]], bool]:
    '''
    Mutual Dependency
    If a candidate a conflicts only with candidate b, and conversely, 
    candidate d conflicts only candidate c, the conflict is resolved by 
    selecting the non-conflicting pair with the lower readability score.

    Parameters
    ----------
    label_candidates : Dict[int, List[LabelCandidate]],
        active label candidates
    conflicts : Dict[Tuple[int, AnchorType], List[Tuple[int, LabelCandidate]]]
        conflicting label candidates

    Returns
    -------
    filtered_candidates, changed : Tuple[Dict[int, List[LabelCandidate]], bool]
        reamining label candidates, wether the set of candidates changed
    '''
    filtered_candidates = copy.deepcopy(label_candidates)
    resolved: Set[int] = set()
    changed = False

    for lid_a, cands_a in filtered_candidates.items():
        if lid_a in resolved or len(filtered_candidates[lid_a]) <= 1:
            continue

        for cand_a in cands_a:
            conflicts_a = conflicts[(lid_a, cand_a.anchor.anchor_type)]
            if not conflicts_a:
                continue

            conflicting_lids = {c_lid for c_lid, _ in conflicts_a}
            if len(conflicting_lids) != 1:
                continue

            lid_b = next(iter(conflicting_lids))
            if lid_b in resolved or lid_b not in label_candidates:
                continue

            cands_b = label_candidates[lid_b]
            for cand_b in cands_b:
                conflicts_b = conflicts[(lid_b, cand_b.anchor.anchor_type)]
                if any(c_lid != lid_a for c_lid, _ in conflicts_b):
                    continue

                valid_pairs = [
                    (ca, cb)
                    for ca in cands_a
                    for cb in cands_b
                    if (lid_b, cb.anchor.anchor_type) not in conflicts[(lid_a, ca.anchor.anchor_type)]
                ]

                if not valid_pairs:
                    break

                valid_pairs.sort(key=lambda pair: (
                    pair[0].anchor.anchor_type.rank + pair[1].anchor.anchor_type.rank
                ))

                best_a, best_b = valid_pairs[0]
                filtered_candidates[lid_a] = [best_a]
                filtered_candidates[lid_b] = [best_b]
                resolved.add(lid_a)
                resolved.add(lid_b)
                changed = True
                break

            if lid_a in resolved:
                break

    return filtered_candidates, changed

def apply_L3(
        label_candidates: Dict[int, List[LabelCandidate]],
        conflicts: Dict[Tuple[int, AnchorType], List[Tuple[int, LabelCandidate]]]
) -> Tuple[Dict[int, List[LabelCandidate]], bool]:
    '''
    Clique Resolution
    If a label has only one remaining candidate a and all its
    conflicting candidates form a clique, i.e., they all
    conflict with each other, then candidate a is selected.

    Parameters
    ----------
    label_candidates : Dict[int, List[LabelCandidate]],
        active label candidates
    conflicts : Dict[Tuple[int, AnchorType], List[Tuple[int, LabelCandidate]]]
        conflicting label candidates

    Returns
    -------
    filtered_candidates, changed : Tuple[Dict[int, List[LabelCandidate]], bool]
        reamining label candidates, wether the set of candidates changed
    '''
    filtered_candidates = copy.deepcopy(label_candidates)
    changed = False

    for lid_a, cands_a in filtered_candidates.items():  # use filtered, not original
        if len(cands_a) != 1:
            continue

        cand_a = cands_a[0]
        conflicts_a = conflicts[(lid_a, cand_a.anchor.anchor_type)]

        if not conflicts_a:
            continue

        conflicting_pairs = [
            (lid_c, cand_c)
            for lid_c, cand_c in conflicts_a
            if lid_c in filtered_candidates
            and cand_c in filtered_candidates[lid_c]
        ]

        if not conflicting_pairs:
            continue

        is_clique = all(
            (lid_j, cand_j) in conflicts[(lid_i, cand_i.anchor.anchor_type)]
            for i, (lid_i, cand_i) in enumerate(conflicting_pairs)
            for j, (lid_j, cand_j) in enumerate(conflicting_pairs)
            if i != j
        )

        if not is_clique:
            continue

        for lid_c, cand_c in conflicting_pairs:
            remaining = [c for c in filtered_candidates[lid_c] if c != cand_c]
            if remaining:
                filtered_candidates[lid_c] = remaining
                changed = True

    return filtered_candidates, changed