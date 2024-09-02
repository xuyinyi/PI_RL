import os.path
import multiprocessing
from queue import Empty
import pandas as pd
from tqdm import tqdm
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import BRICS
from collections import Counter, defaultdict


def process_molecule(queue, mol):
    try:
        pieces = BRICS.BRICSDecompose(mol)
        queue.put(pieces)
    except Exception as e:
        queue.put(e)


def split_polymer(smiles):
    Groups_Counter, Count = Counter(), defaultdict(int)
    for smile in tqdm(smiles):
        mol = Chem.MolFromSmiles(smile)

        smile_new = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smile_new)

        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=process_molecule, args=(queue, mol))
        process.start()
        process.join(timeout=2)

        if process.is_alive():
            process.terminate()
            print(f"Processing of molecule {smile} timed out. Skipping.")
            continue
        try:
            result = queue.get(timeout=1)
            Groups_Counter.update(result)
            if smile not in Count:
                Count[smile] += 1
        except Empty:
            print(f"No result received for molecule {smile} within the timeout. Skipping.")

    return Groups_Counter, Count


def joint_blocks():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    blocks = pd.read_csv(os.path.join(base_dir, 'outputs/building_blocks/blocks_dianhydride_origin.csv'))
    blocks_pubchem = pd.read_csv(os.path.join(base_dir, 'outputs/building_blocks/blocks_dianhydride_pubchem.csv'))
    for block, count in zip(blocks["block"], blocks["count"]):
        if block in blocks_pubchem["block"].values:
            index = blocks_pubchem["block"].values.tolist().index(block)
            blocks_pubchem.loc[index, "count"] += count
        else:
            block_series = pd.Series([block], name="block")
            count_series = pd.Series([count], name="count")
            blocks_pubchem["block"].append(block_series, ignore_index=True)
            blocks_pubchem["count"].append(count_series, ignore_index=True)
    blocks_pubchem.to_csv(os.path.join(base_dir, 'outputs/building_blocks/blocks_dianhydride.csv'), index=False)


if __name__ == "__main__":
    joint_blocks()

