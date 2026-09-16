from typing import Dict, List, Set, Tuple

import networkx as nx

class GraphLattice:
    '''
    Lattice structure, positions, and node labels loaded directly from a
    graphml file (nodes carry `x`/`y` data and optionally `general` data as a
    plain string and `extent`/`intent` data as `;`-separated strings; edges
    are the cover relations, directed from parent to child).
    '''

    def __init__(self, path: str):
        '''
        Parameters
        ----------
        path : str
            Path to the graphml file.
        '''
        self._G = nx.read_graphml(path, node_type=int)
        self.nodes: List[int] = list(self._G.nodes)
        self.positions: Dict[int, Tuple[float, float]] = {
            nid: (data['x'], data['y']) for nid, data in self._G.nodes(data=True)
        }

    def cover_relations(self) -> List[Tuple[int, int]]:
        '''
        Returns
        -------
        cover_relations : List[Tuple[int, int]]
            Edges of the Hasse diagram, directed from parent to child.
        '''
        return list(self._G.edges)

    def children(self, nid: int) -> List[int]:
        '''
        Parameters
        ----------
        nid : int
            Node id.

        Returns
        -------
        children : List[int]
            Direct child node ids.
        '''
        return list(self._G.successors(nid))

    def parents(self, nid: int) -> List[int]:
        '''
        Parameters
        ----------
        nid : int
            Node id.

        Returns
        -------
        parents : List[int]
            Direct parent node ids.
        '''
        return list(self._G.predecessors(nid))

    def general(self, nid: int) -> str:
        '''
        Parameters
        ----------
        nid : int
            Node id.

        Returns
        -------
        general : str
            General label text for this node.
        '''
        return self._G.nodes[nid].get('general', '')

    def extent(self, nid: int) -> Set[str]:
        '''
        Parameters
        ----------
        nid : int
            Node id.

        Returns
        -------
        extent : Set[str]
            Object names newly introduced at this node.
        '''
        return self._labels(nid, 'extent')

    def intent(self, nid: int) -> Set[str]:
        '''
        Parameters
        ----------
        nid : int
            Node id.

        Returns
        -------
        intent : Set[str]
            Attribute names newly introduced at this node.
        '''
        return self._labels(nid, 'intent')

    def _labels(self, nid: int, key: str) -> Set[str]:
        raw = self._G.nodes[nid].get(key, '')
        return {s for s in raw.split(';') if s}
