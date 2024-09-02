from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import networkx as nx
from networkx.classes.graph import Graph
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.rdchem import Mol
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

LABEL = "label"
MST_MAX_WEIGHT = 100


def get_mol(smiles: str) -> Optional[Mol]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol


def get_smiles(mol: Mol) -> str:
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


def clean_smiles(smiles: str) -> str:
    for i in range(1, 17):
        smiles = smiles.replace(f'([{i}*])', '([*])').replace(f'[{i}*]', '[*]')
    return smiles


def sanitize(mol):
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        print(e)
        return None
    return mol


def _set_node_label(graph: Graph, node_label: List[str], label=LABEL) -> None:
    for k, v in enumerate(node_label):
        graph.nodes[k][label] = v


def _set_edge_label(
        graph: Graph, edge_label: List[Tuple[int, int, float]], label=LABEL) -> None:
    for edge in edge_label:
        src, dst, weight = edge
        graph.edges[(src, dst)][label] = weight


def mol_to_graph(mol) -> Graph:
    """
    :param mol:
    :return: Graph
    """
    # _mol_with_atom_index(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    node_label = [atom.GetSymbol() for atom in atoms]
    edge_label = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondTypeAsDouble())
        for bond in bonds
    ]

    # add bond information
    # adj = Chem.GetAdjacencyMatrix(mol)
    adj = np.zeros((len(atoms), len(atoms)))
    for src, dst, bond in edge_label:
        adj[src, dst] = bond
        adj[dst, src] = bond

    graph = nx.from_numpy_array(adj)
    _set_node_label(graph, node_label)
    _set_edge_label(graph, edge_label)
    return graph


def mol_from_graph(graph: Graph) -> Optional[Mol]:
    """

    :param graph:
    :return: mol
    """

    # create empty editable mol object
    mol = Chem.RWMol()

    # add atoms to mol and keep track of index
    node_to_idx = {}
    for i, atom in graph.nodes.items():
        a = Chem.Atom(atom[LABEL])
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx

    n_atom = len(graph.nodes)
    # adjacency_matrix = nx.adjacency_matrix(graph).toarray()
    adjacency_matrix = np.zeros((n_atom, n_atom))
    for (src, dst), bond in graph.edges.items():
        adjacency_matrix[src][dst] = bond[LABEL]
        adjacency_matrix[dst][src] = bond[LABEL]

    # add bonds between adjacent atoms
    for ix, row in enumerate(adjacency_matrix):
        for iy, bond in enumerate(row):
            # only traverse half the matrix
            if iy <= ix:
                continue

            # add relevant bond type (there are many more of these)
            if bond == 0.0:
                continue

            elif bond == 1.0:
                bond_type = Chem.rdchem.BondType.SINGLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

            elif bond == 2.0:
                bond_type = Chem.rdchem.BondType.DOUBLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

            elif bond == 3.0:
                bond_type = Chem.rdchem.BondType.TRIPLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

            elif bond == 1.5:
                bond_type = Chem.rdchem.BondType.AROMATIC
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

    # Convert RWMol to Mol object
    return mol.GetMol()


def mol_with_atom_index(mol: Mol) -> Mol:
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp(
            "molAtomMapNumber", str(mol.GetAtomWithIdx(idx).GetIdx())
        )
    return mol


def tree_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []

    cliques = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            cliques.append([a1, a2])

    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques.extend(ssr)

    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)

    # Merge Rings with intersection > 2 atoms
    for i in range(len(cliques)):
        if len(cliques[i]) <= 2:
            continue
        for atom in cliques[i]:
            for j in nei_list[atom]:
                if i >= j or len(cliques[j]) <= 2:
                    continue
                inter = set(cliques[i]) & set(cliques[j])
                if len(inter) > 2:
                    cliques[i].extend(cliques[j])
                    cliques[i] = list(set(cliques[i]))
                    cliques[j] = []

    cliques = [c for c in cliques if len(c) > 0]
    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)

    # Build edges and add singleton cliques
    edges = defaultdict(int)
    for atom in range(n_atoms):
        if len(nei_list[atom]) <= 1:
            continue
        cnei = nei_list[atom]
        bonds = [c for c in cnei if len(cliques[c]) == 2]
        rings = [c for c in cnei if len(cliques[c]) > 4]
        # In general, if len(cnei) >= 3, a singleton should be added, but 1 bond + 2 ring is currently not dealt with.
        if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2):
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1, c2)] = 1
        elif len(rings) > 2:  # Multiple (n>2) complex rings
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1, c2)] = MST_MAX_WEIGHT - 1
        else:
            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1, c2 = cnei[i], cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    if edges[(c1, c2)] < len(inter):
                        edges[(c1, c2)] = len(
                            inter
                        )  # cnei[i] < cnei[j] by construction

    edges = [u + (MST_MAX_WEIGHT - v,) for u, v in edges.items()]
    if len(edges) == 0:
        return cliques, edges

    # Compute Maximum Spanning Tree
    row, col, data = list(zip(*edges))
    n_clique = len(cliques)
    clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
    junc_tree = minimum_spanning_tree(clique_graph)
    row, col = junc_tree.nonzero()
    edges = [(row[i], col[i]) for i in range(len(row))]
    return (cliques, edges)


class MolGraph:
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = get_mol(smiles)
        self.graph = self._mol_to_graph(self.mol)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.smiles})"

    def _mol_to_graph(self, mol):
        if mol is not None:
            return mol_to_graph(mol)
        return None

    def sanitize(self, mol):
        return sanitize(mol)


def sanitize_molgraph(m_graphs: List[MolGraph]):
    graphs = [mol_to_graph(m.mol) for m in m_graphs if m.mol is not None]
    mols = [sanitize(mol_from_graph(g)) for g in graphs]
    smiles = np.unique([get_smiles(m) for m in mols])
    mols = [get_mol(s) for s in smiles]
    return mols
