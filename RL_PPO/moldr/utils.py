import os
import re
import copy
import pickle
import random
import tempfile
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdChemReactions as Reactions

from ray.tune.logger import UnifiedLogger

environs = {
    'L1': '[C;D3]([#0,#6,#7,#8])(=O)',
    #
    # After some discussion, the L2 definitions ("N.pl3" in the original
    # paper) have been removed and incorporated into a (almost) general
    # purpose amine definition in L5 ("N.sp3" in the paper).
    #
    # The problem is one of consistency.
    #    Based on the original definitions you should get the following
    #    fragmentations:
    #      C1CCCCC1NC(=O)C -> C1CCCCC1N[2*].[1*]C(=O)C
    #      c1ccccc1NC(=O)C -> c1ccccc1[16*].[2*]N[2*].[1*]C(=O)C
    #    This difference just didn't make sense to us. By switching to
    #    the unified definition we end up with:
    #      C1CCCCC1NC(=O)C -> C1CCCCC1[15*].[5*]N[5*].[1*]C(=O)C
    #      c1ccccc1NC(=O)C -> c1ccccc1[16*].[5*]N[5*].[1*]C(=O)C
    #
    # 'L2':'[N;!R;!D1;!$(N=*)]-;!@[#0,#6]',
    # this one turned out to be too tricky to define above, so we set it off
    # in its own definition:
    # 'L2a':'[N;D3;R;$(N(@[C;!$(C=*)])@[C;!$(C=*)])]',
    'L3': '[O;D2]-;!@[#0,#6,#1]',
    'L4': '[C;!D1;!$(C=*)]-;!@[#6]',
    # 'L5':'[N;!D1;!$(N*!-*);!$(N=*);!$(N-[!C;!#0])]-[#0,C]',
    'L5': '[N;!D1;!$(N=*);!$(N-[!#6;!#16;!#0;!#1]);!$([N;R]@[C;R]=O)]',
    'L6': '[C;D3;!R](=O)-;!@[#0,#6,#7,#8]',
    'L7a': '[C;D2,D3]-[#6]',
    'L7b': '[C;D2,D3]-[#6]',
    '#L8': '[C;!R;!D1]-;!@[#6]',
    'L8': '[C;!R;!D1;!$(C!-*)]',
    'L9': '[n;+0;$(n(:[c,n,o,s]):[c,n,o,s])]',
    'L10': '[N;R;$(N(@C(=O))@[C,N,O,S])]',
    'L11': '[S;D2](-;!@[#0,#6])',
    'L12': '[S;D4]([#6,#0])(=O)(=O)',
    'L13': '[C;$(C(-;@[C,N,O,S])-;@[N,O,S])]',
    'L14': '[c;$(c(:[c,n,o,s]):[n,o,s])]',
    'L14b': '[c;$(c(:[c,n,o,s]):[n,o,s])]',
    'L15': '[C;$(C(-;@C)-;@C)]',
    'L16': '[c;$(c(:c):c)]',
    'L16b': '[c;$(c(:c):c)]',
}
reactionDefs = (
    # L1
    [
        ('1', '3', '-'),
        ('1', '5', '-'),
        ('1', '10', '-'),
        ('1', '16', '-'),
    ],

    # L3
    [
        ('3', '4', '-'),
        ('3', '13', '-'),
        ('3', '14', '-'),
        ('3', '15', '-'),
        ('3', '16', '-'),
    ],

    # L4
    [
        ('4', '5', '-'),
        ('4', '11', '-'),
    ],

    # L5
    [
        ('5', '12', '-'),
        ('5', '14', '-'),
        ('5', '16', '-'),
        ('5', '13', '-'),
        ('5', '15', '-'),
    ],

    # L6
    [
        ('6', '13', '-'),
        ('6', '14', '-'),
        ('6', '15', '-'),
        ('6', '16', '-'),
    ],

    # L7
    [
        ('7a', '7b', '='),
    ],

    # L8
    [
        ('8', '9', '-'),
        ('8', '10', '-'),
        ('8', '13', '-'),
        ('8', '14', '-'),
        ('8', '15', '-'),
        ('8', '16', '-'),
    ],

    # L9
    [
        ('9', '13', '-'),  # not in original paper
        ('9', '14', '-'),  # not in original paper
        ('9', '15', '-'),
        ('9', '16', '-'),
    ],

    # L10
    [
        ('10', '13', '-'),
        ('10', '14', '-'),
        ('10', '15', '-'),
        ('10', '16', '-'),
    ],

    # L11
    [
        ('11', '13', '-'),
        ('11', '14', '-'),
        ('11', '15', '-'),
        ('11', '16', '-'),
    ],

    # L12
    # none left

    # L13
    [
        ('13', '14', '-'),
        ('13', '15', '-'),
        ('13', '16', '-'),
    ],

    # L14
    [
        ('14', '14', '-'),  # not in original paper
        ('14', '15', '-'),
        ('14', '16', '-'),
    ],

    # L15
    [
        ('15', '16', '-'),
    ],

    # L16
    [
        ('16', '16', '-'),  # not in original paper
    ],)

smartsGps = copy.deepcopy(reactionDefs)
for gp in smartsGps:
    for j, defn in enumerate(gp):
        g1, g2, bnd = defn
        r1 = environs['L' + g1]
        r2 = environs['L' + g2]
        g1 = re.sub('[a-z,A-Z]', '', g1)
        g2 = re.sub('[a-z,A-Z]', '', g2)
        sma = '[$(%s):1]%s;!@[$(%s):2]>>[%s*]-[*:1].[%s*]-[*:2]' % (r1, bnd, r2, g1, g2)
        gp[j] = sma

for gp in smartsGps:
    for defn in gp:
        try:
            t = Reactions.ReactionFromSmarts(defn)
            t.Initialize()
        except Exception:
            print(defn)
            raise

bondMatchers = []
for i, compats in enumerate(reactionDefs):
    tmp = []
    for i1, i2, bType in compats:
        e1 = environs['L%s' % i1]
        e2 = environs['L%s' % i2]
        patt = '[$(%s)]%s;!@[$(%s)]' % (e1, bType, e2)
        patt = Chem.MolFromSmarts(patt)
        tmp.append((i1, i2, bType, patt))
    bondMatchers.append(tmp)

reactions = tuple([[Reactions.ReactionFromSmarts(y) for y in x] for x in smartsGps])
reverseReactions = []
for i, rxnSet in enumerate(smartsGps):
    for j, sma in enumerate(rxnSet):
        rs, ps = sma.split('>>')
        sma = '%s>>%s' % (ps, rs)
        rxn = Reactions.ReactionFromSmarts(sma)
        labels = re.findall(r'\[([0-9]+?)\*\]', ps)
        rxn._matchers = [Chem.MolFromSmiles('[%s*]' % x) for x in labels]
        reverseReactions.append(rxn)


def BRICSDecompose(mol, allNodes=None, minFragmentSize=1, onlyUseReactions=None, silent=True,
                   keepNonLeafNodes=False, singlePass=False, returnMols=False):
    """ returns the BRICS decomposition for a molecule

    >>> from rdkit import Chem
    >>> m = Chem.MolFromSmiles('CCCOCc1cc(c2ncccc2)ccc1')
    >>> res = list(BRICSDecompose(m))
    >>> sorted(res)
    ['[14*]c1ccccn1', '[16*]c1cccc([16*])c1', '[3*]O[3*]', '[4*]CCC', '[4*]C[8*]']

    >>> res = list(BRICSDecompose(m,returnMols=True))
    >>> res[0]
    <rdkit.Chem.rdchem.Mol object ...>
    >>> smis = [Chem.MolToSmiles(x,True) for x in res]
    >>> sorted(smis)
    ['[14*]c1ccccn1', '[16*]c1cccc([16*])c1', '[3*]O[3*]', '[4*]CCC', '[4*]C[8*]']

    nexavar, an example from the paper (corrected):

    >>> m = Chem.MolFromSmiles('CNC(=O)C1=NC=CC(OC2=CC=C(NC(=O)NC3=CC(=C(Cl)C=C3)C(F)(F)F)C=C2)=C1')
    >>> res = list(BRICSDecompose(m))
    >>> sorted(res)
    ['[1*]C([1*])=O', '[1*]C([6*])=O', '[14*]c1cc([16*])ccn1', '[16*]c1ccc(Cl)c([16*])c1', '[16*]c1ccc([16*])cc1', '[3*]O[3*]', '[5*]NC', '[5*]N[5*]', '[8*]C(F)(F)F']

    it's also possible to keep pieces that haven't been fully decomposed:

    >>> m = Chem.MolFromSmiles('CCCOCC')
    >>> res = list(BRICSDecompose(m,keepNonLeafNodes=True))
    >>> sorted(res)
    ['CCCOCC', '[3*]OCC', '[3*]OCCC', '[3*]O[3*]', '[4*]CC', '[4*]CCC']

    >>> m = Chem.MolFromSmiles('CCCOCc1cc(c2ncccc2)ccc1')
    >>> res = list(BRICSDecompose(m,keepNonLeafNodes=True))
    >>> sorted(res)
    ['CCCOCc1cccc(-c2ccccn2)c1', '[14*]c1ccccn1', '[16*]c1cccc(-c2ccccn2)c1', '[16*]c1cccc(COCCC)c1', '[16*]c1cccc([16*])c1', '[3*]OCCC', '[3*]OC[8*]', '[3*]OCc1cccc(-c2ccccn2)c1', '[3*]OCc1cccc([16*])c1', '[3*]O[3*]', '[4*]CCC', '[4*]C[8*]', '[4*]Cc1cccc(-c2ccccn2)c1', '[4*]Cc1cccc([16*])c1', '[8*]COCCC']

    or to only do a single pass of decomposition:

    >>> m = Chem.MolFromSmiles('CCCOCc1cc(c2ncccc2)ccc1')
    >>> res = list(BRICSDecompose(m,singlePass=True))
    >>> sorted(res)
    ['CCCOCc1cccc(-c2ccccn2)c1', '[14*]c1ccccn1', '[16*]c1cccc(-c2ccccn2)c1', '[16*]c1cccc(COCCC)c1', '[3*]OCCC', '[3*]OCc1cccc(-c2ccccn2)c1', '[4*]CCC', '[4*]Cc1cccc(-c2ccccn2)c1', '[8*]COCCC']

    setting a minimum size for the fragments:

    >>> m = Chem.MolFromSmiles('CCCOCC')
    >>> res = list(BRICSDecompose(m,keepNonLeafNodes=True,minFragmentSize=2))
    >>> sorted(res)
    ['CCCOCC', '[3*]OCC', '[3*]OCCC', '[4*]CC', '[4*]CCC']
    >>> m = Chem.MolFromSmiles('CCCOCC')
    >>> res = list(BRICSDecompose(m,keepNonLeafNodes=True,minFragmentSize=3))
    >>> sorted(res)
    ['CCCOCC', '[3*]OCC', '[4*]CCC']
    >>> res = list(BRICSDecompose(m,minFragmentSize=2))
    >>> sorted(res)
    ['[3*]OCC', '[3*]OCCC', '[4*]CC', '[4*]CCC']


    """
    global reactions
    mSmi = Chem.MolToSmiles(mol, 1)

    if allNodes is None:
        allNodes = set()

    if mSmi in allNodes:
        return set()

    allsub = []
    activePool = {mSmi: mol}
    allNodes.add(mSmi)
    foundMols = {mSmi: mol}
    for gpIdx, reactionGp in enumerate(reactions):
        print(gpIdx)
        newPool = {}
        while activePool:
            matched = False
            nSmi = next(iter(activePool))
            print("nSmi", nSmi)
            mol = activePool.pop(nSmi)
            for rxnIdx, reaction in enumerate(reactionGp):
                if onlyUseReactions and (gpIdx, rxnIdx) not in onlyUseReactions:
                    continue
                if not silent:
                    print('--------')
                    print(smartsGps[gpIdx][rxnIdx])
                ps = reaction.RunReactants((mol,))
                if ps:
                    print("successful")
                    if not silent:
                        print(nSmi, '->', len(ps), 'products')
                    for prodSeq in ps:
                        seqOk = True
                        # we want to disqualify small fragments, so sort the product sequence by size
                        tSeq = [(prod.GetNumAtoms(onlyExplicit=True), idx)
                                for idx, prod in enumerate(prodSeq)]
                        tSeq.sort()
                        for nats, idx in tSeq:
                            prod = prodSeq[idx]
                            try:
                                Chem.SanitizeMol(prod)
                            except Exception:
                                continue
                            pSmi = Chem.MolToSmiles(prod, 1)
                            if minFragmentSize > 0:
                                nDummies = pSmi.count('*')
                                if nats - nDummies < minFragmentSize:
                                    seqOk = False
                                    break
                            prod.pSmi = pSmi
                        ts = [(x, prodSeq[y]) for x, y in tSeq]
                        prodSeq = ts
                        if seqOk:
                            matched = True
                            for nats, prod in prodSeq:
                                pSmi = prod.pSmi
                                # print('\t',nats,pSmi)
                                if pSmi not in allNodes:
                                    if not singlePass:
                                        activePool[pSmi] = prod
                                    allNodes.add(pSmi)
                                    foundMols[pSmi] = prod
                            print(activePool.keys())
            if singlePass or keepNonLeafNodes or not matched:
                newPool[nSmi] = mol
        activePool = newPool
    if not (singlePass or keepNonLeafNodes):
        if not returnMols:
            res = set(activePool.keys())
            res = allsub
        else:
            res = activePool.values()
    else:
        if not returnMols:
            res = allNodes
        else:
            res = foundMols.values()
    return res


dummyPattern = Chem.MolFromSmiles('[*]')


def BRICSBuild(fragments, onlyCompleteMols=True, seeds=None, uniquify=True, scrambleReagents=True,
               maxDepth=3):
    seen = set()
    if not seeds:
        seeds = list(fragments)
    if scrambleReagents:
        seeds = list(seeds)
        random.shuffle(seeds, random=random.random)
    if scrambleReagents:
        tempReactions = list(reverseReactions)
        random.shuffle(tempReactions, random=random.random)
    else:
        tempReactions = reverseReactions
    for seed in seeds:
        seedIsR1 = False
        seedIsR2 = False
        nextSteps = []
        for rxn in tempReactions:
            if seed.HasSubstructMatch(rxn._matchers[0]):
                seedIsR1 = True
            if seed.HasSubstructMatch(rxn._matchers[1]):
                seedIsR2 = True
            for fragment in fragments:
                if not len(set([Chem.MolToSmiles(fragment) for fragment in fragments])) == 1 and Chem.MolToSmiles(
                        fragment) == Chem.MolToSmiles(seed):
                    continue
                ps = None
                if fragment.HasSubstructMatch(rxn._matchers[0]):
                    if seedIsR2:
                        ps = rxn.RunReactants((fragment, seed))
                if fragment.HasSubstructMatch(rxn._matchers[1]):
                    if seedIsR1:
                        ps = rxn.RunReactants((seed, fragment))
                if ps:
                    for p in ps:
                        if uniquify:
                            pSmi = Chem.MolToSmiles(p[0], True)
                            if pSmi in seen:
                                continue
                            else:
                                seen.add(pSmi)
                        if p[0].HasSubstructMatch(dummyPattern):
                            nextSteps.append(p[0])
                            if not onlyCompleteMols:
                                yield p[0]
                        else:
                            yield p[0]
        if nextSteps and maxDepth > 0:
            for p in BRICSBuild(fragments, onlyCompleteMols=onlyCompleteMols, seeds=nextSteps,
                                uniquify=uniquify, maxDepth=maxDepth - 1,
                                scrambleReagents=scrambleReagents):
                if uniquify:
                    pSmi = Chem.MolToSmiles(p, True)
                    if pSmi in seen:
                        continue
                    else:
                        seen.add(pSmi)
                yield p


def save(fpath, obj):
    if not isinstance(fpath, Path):
        fpath = Path(fpath)
    if not fpath.is_dir():
        fpath.parent.mkdir(parents=True, exist_ok=True)
    with fpath.open("wb") as f:
        pickle.dump(obj, f)


def load(fpath):
    if not isinstance(fpath, Path):
        fpath = Path(fpath)
    with fpath.open("rb") as f:
        obj = pickle.load(f)
    return obj


def custom_log_creator(custom_path, custom_str):
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = f"{custom_str}_{timestr}"

    def logger_creator(config):
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


def mapper(n_jobs):
    '''
    Returns function for map call.
    If n_jobs = 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool object, will return its map function
    '''
    if n_jobs == 1:
        def _mapper(*args, **kwargs):
            return list(map(*args, **kwargs))

        return _mapper
    if isinstance(n_jobs, int):
        pool = Pool(n_jobs)

        def _mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate()
            return result

        return _mapper
    return n_jobs.map


def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol


def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def get_morgan_fingerprint(smiles_or_mol, r=6, nBits=1024):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=r, nBits=nBits)
    return fingerprint
