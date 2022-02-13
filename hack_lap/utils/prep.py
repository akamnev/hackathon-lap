import networkx as nx
from hack_lap.utils import ATOM2IDX, BondType


def clean_graph(mol):
    nodes = [
        (i, {'id': a.GetAtomicNum()})
        for i, a in enumerate(mol.GetAtoms())
    ]
    edges = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx(),
         {'bond_type': str(b.GetBondType())})
        for b in mol.GetBonds()
    ]
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    if not nx.is_connected(g):
        sg = [
            c for c in
            sorted(nx.connected_components(g), key=len, reverse=True)
        ]
        sg = sg[1:]
        for si in sg:
            g.remove_nodes_from(si)
        g = nx.convert_node_labels_to_integers(g)

    nodes = [ATOM2IDX[g.nodes[i]['id']] for i in g]
    edges = {(i, j, g.edges[i, j]['bond_type']) for i, j in g.edges}
    edges.update({(j, i, g.edges[i, j]['bond_type']) for i, j in g.edges})
    edges = sorted(list(edges), key=lambda x: x[:-1])
    edges_attr = [int(BondType[v[-1]]) for v in edges]
    edges = [v[:-1] for v in edges]
    return nodes, edges, edges_attr
