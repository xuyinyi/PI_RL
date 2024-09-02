import random
from rdkit.Chem import AllChem as Chem


def reaction(mol_dianhydride, mol_diamine):
    """The dianhydride reacts with diamine to form PI"""
    reactionTemplate = Chem.ReactionFromSmarts(
        '[#8:1]=[#6:2][#8][#6:3]=[#8:4].[#6:5]-[#7:6]>>[#6:5]-[#7:6]([#6:3]=[#8:4])[#6:2]=[#8:1]')
    PI = reactionTemplate.RunReactants((mol_dianhydride, mol_diamine))
    product = []
    for pi in PI:
        pi_smi = Chem.MolToSmiles(pi[0])
        if Chem.MolFromSmiles(pi_smi):
            product.append(Chem.MolToSmiles(pi[0]))
        else:
            continue
    product = list(set(product))
    # if len(product) > 1:
    #     product = [random.choice(product)]
    return product


def replace_reaction(mol_PI):
    """Modify the generated PI molecule"""
    reactionTemplate = Chem.ReactionFromSmarts(
        '[#8:1]=[#6:2][#8][#6:3]=[#8:4]>>[#6]-[#7]([#6:3]=[#8:4])[#6:2]=[#8:1]')
    PI = reactionTemplate.RunReactants((mol_PI,))
    product = []
    for pi in PI:
        pi_smi = Chem.MolToSmiles(pi[0])
        if Chem.MolFromSmiles(pi_smi):
            product.append(Chem.MolToSmiles(pi[0]))
        else:
            continue
    product = list(set(product))
    if len(product) != 0:
        _SMILES = random.choice(product)
        return _SMILES
    else:
        return None


def generate_PI(smiles_dianhydride, smiles_diamine):
    PI_list = list()
    mol_dianhydride = Chem.MolFromSmiles(smiles_dianhydride)
    mol_diamine = Chem.MolFromSmiles(smiles_diamine)
    PI = reaction(mol_dianhydride, mol_diamine)
    if len(PI) != 0:
        patt1 = Chem.MolFromSmarts('[N;H2]')
        replace_1 = Chem.MolFromSmarts('C')
        for pi in PI:
            mols = []
            mols.extend(Chem.ReplaceSubstructs(Chem.MolFromSmiles(pi), patt1, replace_1))
            pi = [Chem.MolToSmiles(mol) for mol in mols][0]
            pi = replace_reaction(Chem.MolFromSmiles(pi))
            if pi:
                mol = Chem.MolFromSmiles(pi)
                idx = [idx for idx, atomnum in enumerate([atom.GetAtomicNum() for atom in mol.GetAtoms()]) if
                       atomnum == 0]
                [mol.GetAtomWithIdx(i).SetAtomicNum(1) for i in idx]
                smile = Chem.MolToSmiles(mol)
                if smile not in PI_list:
                    PI_list.append(smile)
                else:
                    continue
            else:
                continue
        if len(PI_list) != 0:
            res = random.choice(PI_list)
            return res
        else:
            return None
    else:
        return None
