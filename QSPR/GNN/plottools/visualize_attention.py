import os
import io
import json
import math
import itertools
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from rdkit import Chem
from rdkit.Chem import rdDepictor, AllChem, rdCoordGen, rdGeometry
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
from load_data import load_FraGAT, load_Attention, load_Pred


def visualize_atom_weights(path, smiles_list, targets_list, tag_list, predictions_list, attentions_list_array,
                           time_step):
    save_path = os.path.realpath('.//library/' + path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    assert time_step < attentions_list_array[0].shape[0], 'Unexpected id for readout step.'
    dict = {'SMILE': [], 'Target': [], 'Prediction': [], 'Residual': [], 'Time_step': [], 'Tag': []}
    for i, smi in enumerate(smiles_list):
        dict['SMILE'].append(smi)
        dict['Target'].append(targets_list[i])
        dict['Prediction'].append(predictions_list[i])
        dict['Time_step'].append(time_step)
        dict['Tag'].append(tag_list[i])
        attention = attentions_list_array[i]
        attention = np.asarray(attention, dtype=np.float64)
        if not math.isnan(targets_list[i]):
            dict['Residual'].append(abs(predictions_list[i] - targets_list[i]))
        else:
            dict['Residual'].append(np.nan)
        visualize_atom_weights_single(save_path, smi, i, attention, time_step)
    df = pd.DataFrame.from_dict(dict)
    df.to_csv(save_path + '/' + 'config.csv', index=False)


def visualize_frag_weights(dataset, smiles_list, targets_list, predictions_list, attentions_list_array, atom_mask_list,
                           time_step):
    save_path = os.path.realpath('.//library/figure/' + dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(attentions_list_array[0].shape[0])
    assert time_step < attentions_list_array[0].shape[0], 'Unexpected id for readout step.'
    dict = {'SMILE': [], 'Target': [], 'Prediction': [], 'Residual': [], 'Time_step': []}
    for i, smi in enumerate(smiles_list):
        dict['SMILE'].append(smi)
        dict['Target'].append(targets_list[i])
        dict['Prediction'].append(predictions_list[i])
        dict['Time_step'].append(time_step)
        attention = attentions_list_array[i]
        attention = np.asarray(attention, dtype=np.float64)
        atom_mask = atom_mask_list[i]
        i_frag, _ = atom_mask.nonzero()
        if not math.isnan(targets_list[i]):
            dict['Residual'].append(abs(predictions_list[i] - targets_list[i]))
        else:
            dict['Residual'].append(np.nan)
        if np.max(i_frag) < attentions_list_array[0].shape[0]:
            visualize_frag_weights_single(save_path, smi, i, attention, atom_mask, time_step)
    df = pd.DataFrame.from_dict(dict)
    df.to_csv(save_path + '/' + 'config.csv', index=False)


def visualize_frag_weights_pred(dataset, smiles_list, predictions_list, attentions_list_array, atom_mask_list,
                                time_step):
    save_path = os.path.realpath('.//library/figure/' + dataset + '/pred/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    assert time_step < attentions_list_array[0].shape[1], 'Unexpected id for readout step.'
    dict = {'SMILE': [], 'Prediction': [], 'Time_step': []}
    for i, smi in enumerate(smiles_list):
        dict['SMILE'].append(smi)
        dict['Prediction'].append(predictions_list[i])
        dict['Time_step'].append(time_step)
        attention = attentions_list_array[i]
        attention = np.asarray(attention, dtype=np.float64)
        atom_mask = atom_mask_list[i]
        i_frag, _ = atom_mask.nonzero()
        # if np.max(i_frag) <= attentions_list_array[9].shape[0]:
        visualize_frag_weights_single(save_path, smi, i, attention, atom_mask, time_step)
    df = pd.DataFrame.from_dict(dict)
    df.to_csv(save_path + 'config.csv', index=False)


def visualize_atom_weights_single_back(path, smile, idx, attention, time_step):
    """Visualize weights of atoms
    Parameters
    ----------
    smiles : str
        SMILEs of single molecule.
    names : str
        Name of single molecule.
    params : dict
        Set of parameters for the workflow.
    attention : ndarray
        Attention weights of atoms. [V, ]
    error : float
        Relative error of prediction from target.
    time_step : int
        Where to extract attention weights, which layer in readout.
    """
    mol = Chem.MolFromSmiles(smile)

    max_value = max(attention)
    min_value = min(attention)
    mean_value = np.nanmean(attention)

    # frag_weights = (frag_weights - min_value) / (max_value - min_value)
    if max_value != min_value:
        atom_weights = [(x - min_value) / (max_value - min_value) for x in attention]
    # color the atoms based on fragments:
    # atom_mask: size: [idx_fragments, idx_nodes]
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, atom_weights, colorMap=matplotlib.cm.Reds)

    save_name = '{}_{}'.format(idx, int(time_step)) + '.tif'
    # drawer.save(save_path, bbox_inches='tight')
    # drawer.save(save_path)
    # plt.close(fig)
    save_path = path + '/' + save_name
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def visualize_atom_weights_single(path, smile, idx, attention, time_step):
    """Visualize weights of atoms
    Parameters
    ----------
    smiles : str
        SMILEs of single molecule.
    names : str
        Name of single molecule.
    params : dict
        Set of parameters for the workflow.
    attention : ndarray
        Attention weights of atoms. [V, ]
    error : float
        Relative error of prediction from target.
    time_step : int
        Where to extract attention weights, which layer in readout.
    """
    mol = Chem.MolFromSmiles(smile)

    max_value = max(attention)
    min_value = min(attention)
    mean_value = np.nanmean(attention)

    if max_value != min_value:
        atom_weights = [(x - min_value) / (max_value - min_value) for x in attention]

    # atom_mask: size: [idx_fragments, idx_nodes]

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.28)
    cmap = cm.get_cmap('viridis_r')
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
    atom_colors = {i: plt_colors.to_rgba(atom_weights[i]) for i in range(mol.GetNumAtoms())}

    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(800, 400)
    opts = rdMolDraw2D.MolDrawOptions()
    opts.SetFontSize = -1
    opts.bondLineWidth = 6
    opts.multipleBondOffset = 0.15
    opts.highlightBondWidthMultiplier = 6
    opts.padding = 0
    opts.useBWAtomPalette()  # set all atoms colour to black
    opts.splitBonds = False
    opts.singleColourWedgeBonds = True  # wedged and dashed bonds color use symbol colour instead of inheriting from atoms
    drawer.SetDrawOptions(opts)

    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(mol, highlightAtoms=range(mol.GetNumAtoms()), highlightBonds=[],
                        highlightAtomColors=atom_colors)
    drawer.FinishDrawing()

    save_name = '{}_{}'.format(idx, int(time_step))
    # drawer.save(save_path, bbox_inches='tight')
    # drawer.save(save_path)
    # plt.close(fig)
    # img = Image.open(io.BytesIO(drawer.GetDrawingText()))
    save_path = path + '/' + save_name
    # img.save(save_path)
    # drawer.WriteDrawingText(save_path)
    with open(save_path + '.svg', 'w+') as output:
        output.write(drawer.GetDrawingText())


def visualize_frag_weights_single_back2(path, smile, idx, frag_attention, atom_mask, time_step):
    """Visualize weights of atoms
    Parameters
    ----------
    smiles : str
        SMILEs of single molecule.
    names : str
        Name of single molecule.
    params : dict
        Set of parameters for the workflow.
    attention : ndarray
        Attention weights of atoms. [V, ]
    error : float
        Relative error of prediction from target.
    time_step : int
        Where to extract attention weights, which layer in readout.
    """
    mol = Chem.MolFromSmiles(smile)

    max_value = max(frag_attention)
    min_value = min(frag_attention)
    mean_value = np.nanmean(frag_attention)

    # frag_weights = (frag_weights - min_value) / (max_value - min_value)
    if max_value != min_value:
        frag_weights = [(x - min_value) / (max_value - min_value) for x in frag_attention]

    atom_weights = np.zeros((mol.GetNumAtoms(), 1))
    i_frag, i_atom = atom_mask.nonzero()
    for i, j in zip(i_frag, i_atom):
        atom_weights[j] = frag_weights[i]
    # color the atoms based on fragments:
    # atom_mask: size: [idx_fragments, idx_nodes]

    atom_weights = atom_weights.flatten()
    fig = Chem.Draw.MolToMPL(mol, coordScale=1.5, size=(250, 250))

    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
    atom_colors = {i: plt_colors.to_rgba(atom_weights[i]) for i in range(mol.GetNumAtoms())}

    rdDepictor.Compute2DCoords(mol)

    locs = []
    for i in range(mol.GetNumAtoms()):
        p = mol.GetConformer().GetAtomPosition(i)
        locs.append(rdGeometry.Point2D(p.x, p.y))

    bond = mol.GetBondWithIdx(0)
    idx1 = bond.GetBeginAtomIdx()
    idx2 = bond.GetEndAtomIdx()
    sigma = 0.3 * math.sqrt(sum([(mol._atomPs[idx1][i] - mol._atomPs[idx2][i]) ** 2 for i in range(2)]))
    sigma = round(sigma, 2)
    x, y, z = Chem.Draw.calcAtomGaussians(mol, sigma, weights=atom_weights, step=0.1)

    fig.axes[0].imshow(z, cmap=matplotlib.cm.Reds, interpolation='bilinear', origin='lower',
                       extent=(0, 1, 0, 1), vmin=0, vmax=1)

    ps = Chem.Draw.ContourParams()
    ps.fillGrid = True
    ps.gridResolution = 0.1
    ps.extraGridPadding = 0.5

    drawer = rdMolDraw2D.MolDraw2DSVG(800, 400)
    # Chem.Draw.ContourAndDrawGaussians(drawer, locs, atom_weights, sigmas, nContours=1, params=ps)
    # drawer.SetFontSize(10)
    mol = rdMolDraw2D.PrepareMolForDrawing(mol)

    drawer.DrawMolecule(mol, highlightAtoms=range(mol.GetNumAtoms()), highlightBonds=[],
                        highlightAtomColors=atom_colors)
    # drawer.DrawEllipse(locs[0], locs[1])
    drawer.FinishDrawing()
    # svg = drawer.GetDrawingText()
    # svg = svg.replace('svg:', '')

    save_name = '{}_{}'.format(idx, int(time_step))
    # drawer.save(save_path, bbox_inches='tight')
    # drawer.save(save_path)
    # plt.close(fig)
    # img = Image.open(io.BytesIO(drawer.GetDrawingText()))
    save_path = path + '/' + save_name
    # img.save(save_path)
    # drawer.WriteDrawingText(save_path)
    with open(save_path + '.svg', 'w+') as output:
        output.write(drawer.GetDrawingText())


def visualize_frag_weights_single_back(path, smile, idx, frag_attention, atom_mask, time_step):
    mol = Chem.MolFromSmiles(smile)

    max_value = max(frag_attention)
    min_value = min(frag_attention)
    mean_value = np.nanmean(frag_attention)

    # frag_weights = (frag_weights - min_value) / (max_value - min_value)
    if max_value != min_value:
        frag_weights = [(x - min_value) / (max_value - min_value) for x in frag_attention]

    atom_weights = np.zeros((mol.GetNumAtoms(), 1))
    i_frag, i_atom = atom_mask.nonzero()
    for i, j in zip(i_frag, i_atom):
        atom_weights[j] = frag_weights[i]
    # color the atoms based on fragments:
    # atom_mask: size: [idx_fragments, idx_nodes]

    atom_weights = atom_weights.flatten()

    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, atom_weights, colorMap='bwr', step=0.001, contourLines=0)
    save_name = '{}_{}'.format(idx, int(time_step)) + '.tiff'
    save_path = path + '/' + save_name
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def visualize_frag_weights_single(path, smile, idx, frag_attention, atom_mask, time_step):
    mol = Chem.MolFromSmiles(smile)

    frag_attention = frag_attention[:, time_step]
    max_value = max(frag_attention)
    min_value = min(frag_attention)
    mean_value = np.nanmean(frag_attention)

    if max_value != min_value:
        frag_weights = [(x - min_value) / (max_value - min_value) for x in frag_attention]
    else:
        frag_weights = [x for x in frag_attention]
    atom_weights = np.zeros((mol.GetNumAtoms(), 1))
    bond_weights = np.zeros((mol.GetNumBonds(), 1))
    i_frag, i_atom = atom_mask.nonzero()  # atom_mask, 0-1 matrix, loc index refers to the (frag_id, atom_id)

    for i, j in zip(i_frag, i_atom):
        atom_weights[j] = frag_weights[i]
    # number of fragments
    all_bond_set = set(range(mol.GetNumBonds()))  # set of all bond idx
    frag_bond_set = set()  # set of bond idx in fragment
    for i in range(atom_mask.shape[0]):
        i_atom = atom_mask[i, :].nonzero()[0]
        # mol.GetAtomWithIdx(int(i_atom[0])).SetProp('atomNote', 'frag_' + str(i))
        if len(i_atom) >= 2:
            for comb in itertools.combinations(i_atom, 2):
                bond = mol.GetBondBetweenAtoms(int(comb[0]), int(comb[1]))
                if bond:
                    i_bond = bond.GetIdx()
                    frag_bond_set.update(set([i_bond]))
                    bond_weights[i_bond] = frag_weights[i]
            # mol.GetBondWithIdx(i_bond).SetProp('bondNote', 'frag_' + str(i))
        # else:
        # mol.GetAtomWithIdx(int(i_atom)).SetProp('atomNote', 'frag_' + str(i))
    brk_bond_set = all_bond_set - frag_bond_set  # set of bond idx between fragments
    # for i in list(brk_bond_set):
    #    bond_weights = np.delete(bond_weights, i, axis=0)
    # color the atoms based on fragments:
    # atom_mask: size: [idx_fragments, idx_nodes]

    atom_weights = atom_weights.flatten()
    bond_weights = bond_weights.flatten()
    norm = matplotlib.colors.Normalize(vmin=-0.1, vmax=1.1)
    colors_255 = [(105, 17, 115), (209, 114, 129), (245, 240, 175)]
    colors = [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in colors_255]
    cmap_name = 'my_list'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=500)
    # cmap = cm.get_cmap('inferno')
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
    atom_colors = {i: plt_colors.to_rgba(atom_weights[i]) for i in range(mol.GetNumAtoms())}
    bond_colors = {i: plt_colors.to_rgba(bond_weights[i]) for i in list(frag_bond_set)}
    rdDepictor.Compute2DCoords(mol)

    locs = []
    for i in range(mol.GetNumAtoms()):
        p = mol.GetConformer().GetAtomPosition(i)
        locs.append(rdGeometry.Point2D(p.x, p.y))

    bond = mol.GetBondWithIdx(0)
    idx1 = bond.GetBeginAtomIdx()
    idx2 = bond.GetEndAtomIdx()
    sigma = 0.3 * (mol.GetConformer().GetAtomPosition(idx1) -
                   mol.GetConformer().GetAtomPosition(idx2)).Length()
    sigma = round(sigma, 2)
    sigmas = [sigma] * mol.GetNumAtoms()

    ps = Chem.Draw.ContourParams()
    ps.fillGrid = True
    ps.gridResolution = 0.1
    ps.extraGridPadding = 0.5

    drawer = rdMolDraw2D.MolDraw2DSVG(800, 400)
    opts = rdMolDraw2D.MolDrawOptions()
    opts.SetFontSize = -1
    opts.bondLineWidth = 5
    opts.multipleBondOffset = 0.15
    opts.highlightBondWidthMultiplier = 6
    opts.padding = 0
    opts.useBWAtomPalette()  # set all atoms colour to black
    opts.splitBonds = False
    opts.singleColourWedgeBonds = True  # wedged and dashed bonds color use symbol colour instead of inheriting from atoms
    drawer.SetDrawOptions(opts)
    # opts.setSymbolColour(defaultdict(lambda : (0, 0, 0)))
    mol = rdMolDraw2D.PrepareMolForDrawing(mol)

    drawer.DrawMolecule(mol, highlightAtoms=range(mol.GetNumAtoms()), highlightBonds=list(frag_bond_set),
                        highlightAtomColors=atom_colors, highlightBondColors=bond_colors)

    drawer.FinishDrawing()

    # svg = drawer.GetDrawingText()
    # svg = svg.replace('svg:', '')

    save_name = '{}_{}'.format(idx, int(time_step))
    # drawer.save(save_path, bbox_inches='tight')
    # drawer.save(save_path)
    # plt.close(fig)
    # img = Image.open(io.BytesIO(drawer.GetDrawingText()))
    save_path = path + '/' + save_name
    # img.save(save_path)
    # drawer.WriteDrawingText(save_path)

    with open(save_path + '.svg', 'w+') as output:
        output.write(drawer.GetDrawingText())
    # fig = plt.imread(save_path + '.svg')
    # fig.display()


def print_frag_attentions(path, smiles_list, attentions_list_array, atom_mask_list, frag_flag_list, time_step):
    save_path = os.path.realpath('.//library/attention/' + path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    assert time_step < attentions_list_array[0].shape[1], 'Unexpected id for readout step.'
    mol_dict = {}
    for i, smi in enumerate(smiles_list):
        mol_dict[str(i)] = {}
        assert atom_mask_list[i].shape[0] == len(frag_flag_list[i]), 'Numbers of fragments are unmatched.'
        # max_att = max(attentions_list_array[i][:, time_step])
        # min_att = min(attentions_list_array[i][:, time_step])
        for j in range(atom_mask_list[i].shape[0]):
            mol_dict[str(i)]['frag_' + str(j)] = {}
            mol_dict[str(i)]['frag_' + str(j)]['frag_name'] = frag_flag_list[i][j]
            x = attentions_list_array[i][j, time_step]
            mol_dict[str(i)]['frag_' + str(j)]['attention_weight'] = float(x)
            # if max_att != min_att:
            # mol_dict[str(smi)]['frag_' + str(j)]['attention_weight'] = float((x - min_att) / (max_att - min_att))
            # else:
            # mol_dict[str(smi)]['frag_' + str(j)]['attention_weight'] = float(x)

    with open(save_path + '/' + 'fragments_info.json', 'w', newline='\n') as f:
        str_ = json.dumps(mol_dict, indent=1)
        f.write(str_)


def print_atom_attentions(path, smiles_list, attentions_list_array, atom_mask_list, frag_flag_list, time_step):
    save_path = os.path.realpath('.//library/' + path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    assert time_step < attentions_list_array[0].shape[1], 'Unexpected id for readout step.'
    mol_dict = {}
    for i, smi in enumerate(smiles_list):
        mol_dict[str(smi)] = {}
        assert atom_mask_list[i].shape[0] == len(frag_flag_list[i]), 'Numbers of fragments are unmatched.'
        # max_att = max(attentions_list_array[i][:, time_step])
        # min_att = min(attentions_list_array[i][:, time_step])
        for j in range(atom_mask_list[i].shape[0]):
            mol_dict[str(smi)]['frag_' + str(j)] = {}
            mol_dict[str(smi)]['frag_' + str(j)]['frag_name'] = frag_flag_list[i][j]
            x = attentions_list_array[i][j, time_step]
            mol_dict[str(smi)]['frag_' + str(j)]['attention_weight'] = float(x)
            # if max_att != min_att:
            # mol_dict[str(smi)]['frag_' + str(j)]['attention_weight'] = float((x - min_att) / (max_att - min_att))
            # else:
            # mol_dict[str(smi)]['frag_' + str(j)]['attention_weight'] = float(x)

    with open(save_path + '/' + 'fragments_info.json', 'w', newline='\n') as f:
        str_ = json.dumps(mol_dict, indent=1)
        f.write(str_)


def analysis_frag_attentions(path):
    MG_plus_reference_path = r"E:\CPI\PI\GNN\datasets\MG_plus_reference.csv"
    read_path = os.path.realpath(f'.//library/attention/{path}/' + 'fragments_info.json')

    save_path = os.path.realpath(f'.//library/attention/{path}/' + 'fragments.csv')
    importance_save_path = os.path.realpath(f'.//library/attention/{path}/' + 'fragments_importance.csv')
    df_all_frag = pd.read_csv(MG_plus_reference_path)
    all_frag = df_all_frag["First-Order Group"].values.tolist()
    priority = df_all_frag["Priority"].values.tolist()
    frag_dict = {}
    frag_importance = {"frag": [], "priority": [], "mean": [], "std": []}
    df = pd.DataFrame()
    with open(read_path, 'r', encoding='utf-8') as f:
        jsondata = f.read()
        jsondict = json.loads(jsondata)
        for smi in jsondict:
            for frag in jsondict[smi]:
                if jsondict[smi][frag]["frag_name"] not in frag_dict:
                    frag_dict[jsondict[smi][frag]["frag_name"]] = [jsondict[smi][frag]["attention_weight"]]
                else:
                    frag_dict[jsondict[smi][frag]["frag_name"]].append(jsondict[smi][frag]["attention_weight"])
    for key, value in frag_dict.items():
        df = pd.concat((df, pd.DataFrame(value, columns=[key])), axis=1)
        frag_importance["frag"].append(key)
        if key in all_frag:
            frag_importance["priority"].append(priority[all_frag.index(key)])
        else:
            frag_importance["priority"].append(19)
        frag_importance["mean"].append(np.mean(value))
        frag_importance["std"].append(np.std(value))
    df.to_csv(save_path, index=False)
    frag_importance_df = pd.DataFrame.from_dict(frag_importance)
    frag_importance_df_sorted = frag_importance_df.sort_values(by='priority')
    frag_importance_df_sorted.to_csv(importance_save_path, index=False)


if __name__ == "__main__":
    dataset_list = ['transmittance(400)', 'cte', 'strength', 'tg']
    # for i in range(4):
    #     dataset, smiles_list, attentions_list_array, atom_mask_list, frag_flag_list, time_step = load_Attention(
    #         dataset_list[i])
    #     print_frag_attentions(dataset, smiles_list, attentions_list_array, atom_mask_list, frag_flag_list, time_step)
    #     analysis_frag_attentions(dataset)

    # op_list = [13, 23, 71, 3]
    # for i in range(4):
    #     dataset, smiles_list, targets_list, predictions_list, attentions_list_array, atom_mask_list, time_step = load_FraGAT(
    #         dataset_list[i], op_list[i])
    #     visualize_frag_weights(dataset, smiles_list, targets_list, predictions_list, attentions_list_array,
    #                            atom_mask_list, time_step)

    op_list = [95, 98, 86, 96]
    for i in range(4):
        dataset, smiles_list, predictions_list, attentions_list_array, atom_mask_list, time_step = load_Pred(
            dataset_list[i], op_list[i])
        visualize_frag_weights_pred(dataset, smiles_list, predictions_list, attentions_list_array, atom_mask_list,
                                    time_step)
