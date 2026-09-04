from __future__ import annotations

import hashlib
import inspect
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import FrozenSet, Sequence, Set, Tuple


@dataclass(frozen=True)
class BlockMetadata:
    action_id: int
    raw_smiles: str
    canonical_smiles: str
    has_attachment: bool
    attachment_labels: FrozenSet[int]
    attachment_count: int
    atom_count: int
    is_complete_for_side: bool


class LegacyDAPiGenChemistryBackend(object):
    """Deterministic implementation of DAPiGen's released chemistry rules.

    BRICS products and final PI products are enumerated and canonicalized. No
    Python/NumPy global RNG and no terminal property score are consulted.
    """

    backend_version = "dapigen-legacy-chemistry-deterministic-v2"

    def __init__(self, allow_rdkit_brics_fallback: bool = False) -> None:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        self.Chem = Chem
        self.AllChem = AllChem
        self.allow_rdkit_brics_fallback = bool(allow_rdkit_brics_fallback)
        self.dianhydride_pattern = Chem.MolFromSmarts(
            "[#8]=[#6]1[#6][#6][#6](=[#8])[#8]1"
        )
        self.diamine_pattern = Chem.MolFromSmarts("[#7H2]")
        self._first_reaction = AllChem.ReactionFromSmarts(
            "[#8:1]=[#6:2][#8][#6:3]=[#8:4]."
            "[#6:5]-[#7:6]>>[#6:5]-[#7:6]([#6:3]=[#8:4])[#6:2]=[#8:1]"
        )
        self._second_reaction = AllChem.ReactionFromSmarts(
            "[#8:1]=[#6:2][#8][#6:3]=[#8:4]>>"
            "[#6]-[#7]([#6:3]=[#8:4])[#6:2]=[#8:1]"
        )
        self._terminal_amine_pattern = Chem.MolFromSmarts("[N;H2]")
        self._carbon_replacement = Chem.MolFromSmarts("C")
        self.compatibility_pairs = self._load_compatibility_pairs()
        (
            self._BRICSBuild,
            self.assembly_backend_name,
            closure_reactions,
        ) = self._resolve_brics_builder()
        self._closure_reactions = self._index_closure_reactions(closure_reactions)
        try:
            import rdkit

            rdkit_version = str(getattr(rdkit, "__version__", "unknown"))
        except Exception:
            rdkit_version = "unknown"
        self.backend_version = (
            "dapigen-legacy-chemistry-deterministic-v2:%s:rdkit-%s"
            % (self.assembly_backend_name, rdkit_version)
        )
        self.diagnostics = defaultdict(int)

    def _load_compatibility_pairs(self) -> FrozenSet[Tuple[int, int]]:
        try:
            from RL_PPO.moldr.utils import reactionDefs
        except Exception:
            # Numerical labels from the released DAPiGen/RDKit BRICS table.
            reaction_defs = (
                ((1, 3), (1, 5), (1, 10), (1, 16)),
                ((3, 4), (3, 13), (3, 14), (3, 15), (3, 16)),
                ((4, 5), (4, 11)),
                ((5, 12), (5, 14), (5, 16), (5, 13), (5, 15)),
                ((6, 13), (6, 14), (6, 15), (6, 16)),
                ((7, 7),),
                ((8, 9), (8, 10), (8, 13), (8, 14), (8, 15), (8, 16)),
                ((9, 13), (9, 14), (9, 15), (9, 16)),
                ((10, 13), (10, 14), (10, 15), (10, 16)),
                ((11, 13), (11, 14), (11, 15), (11, 16)),
                ((13, 14), (13, 15), (13, 16)),
                ((14, 14), (14, 15), (14, 16)),
                ((15, 16),),
                ((16, 16),),
            )
            pairs = set()  # type: Set[Tuple[int, int]]
            for group in reaction_defs:
                for left, right in group:
                    pairs.add((int(left), int(right)))
                    pairs.add((int(right), int(left)))
            return frozenset(pairs)

        pairs = set()  # type: Set[Tuple[int, int]]
        for group in reactionDefs:
            for left, right, _bond in group:
                left_number = int(re.findall(r"\d+", str(left))[0])
                right_number = int(re.findall(r"\d+", str(right))[0])
                pairs.add((left_number, right_number))
                pairs.add((right_number, left_number))
        return frozenset(pairs)

    def canonicalize(self, smiles: str) -> str:
        molecule = self.Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError("Invalid SMILES: %s" % smiles)
        self.Chem.SanitizeMol(molecule)
        return self.Chem.MolToSmiles(molecule, canonical=True)

    def atom_count(self, smiles: str) -> int:
        molecule = self.Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError("Invalid SMILES: %s" % smiles)
        return int(molecule.GetNumAtoms())

    def attachment_labels(self, smiles: str) -> FrozenSet[int]:
        molecule = self.Chem.MolFromSmiles(smiles)
        if molecule is None:
            return frozenset()
        labels = set()
        for atom in molecule.GetAtoms():
            if int(atom.GetAtomicNum()) == 0:
                labels.add(int(atom.GetIsotope()))
        return frozenset(labels)

    def attachment_count(self, smiles: str) -> int:
        molecule = self.Chem.MolFromSmiles(smiles)
        if molecule is None:
            return 0
        return sum(1 for atom in molecule.GetAtoms() if int(atom.GetAtomicNum()) == 0)

    def attachment_label_histogram(
        self, smiles: str, maximum_label: int
    ) -> Sequence[int]:
        """Count dummy-atom isotope labels without erasing reaction semantics."""

        maximum_label = int(maximum_label)
        molecule = self.Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError("Invalid SMILES: %s" % smiles)
        counts = [0] * (maximum_label + 1)
        for atom in molecule.GetAtoms():
            if int(atom.GetAtomicNum()) != 0:
                continue
            label = int(atom.GetIsotope())
            if label < 0 or label > maximum_label:
                raise ValueError(
                    "Attachment label %d exceeds configured range 0..%d."
                    % (label, maximum_label)
                )
            counts[label] += 1
        return tuple(counts)

    def labels_can_react(
        self, base_labels: FrozenSet[int], block_labels: FrozenSet[int]
    ) -> bool:
        if not base_labels or not block_labels:
            return False
        return any(
            (left, right) in self.compatibility_pairs
            for left in base_labels
            for right in block_labels
        )

    def can_react(self, base_smiles: str, block_smiles: str) -> bool:
        return self.labels_can_react(
            self.attachment_labels(base_smiles),
            self.attachment_labels(block_smiles),
        )

    def is_complete_dianhydride(self, smiles: str) -> bool:
        molecule = self.Chem.MolFromSmiles(smiles)
        return bool(
            molecule is not None
            and self.attachment_count(smiles) == 0
            and len(molecule.GetSubstructMatches(self.dianhydride_pattern)) == 2
        )

    def is_complete_diamine(self, smiles: str) -> bool:
        molecule = self.Chem.MolFromSmiles(smiles)
        return bool(
            molecule is not None
            and self.attachment_count(smiles) == 0
            and len(molecule.GetSubstructMatches(self.diamine_pattern)) == 2
        )

    def block_metadata(self, action_id: int, smiles: str, side: str) -> BlockMetadata:
        canonical = self.canonicalize(smiles)
        labels = self.attachment_labels(canonical)
        if side == "dianhydride":
            complete = self.is_complete_dianhydride(canonical)
        elif side == "diamine":
            complete = self.is_complete_diamine(canonical)
        else:
            raise ValueError("Unknown side: %s" % side)
        return BlockMetadata(
            action_id=int(action_id),
            raw_smiles=smiles,
            canonical_smiles=canonical,
            has_attachment=bool(labels),
            attachment_labels=labels,
            attachment_count=self.attachment_count(canonical),
            atom_count=self.atom_count(canonical),
            is_complete_for_side=bool(complete),
        )

    @staticmethod
    def _implementation_fingerprint(function) -> str:
        """Hash the loaded BRICS implementation, not only its import name."""

        digest = hashlib.sha256()
        source_path = inspect.getsourcefile(function)
        if source_path:
            path = str(source_path)
            try:
                with open(path, "rb") as handle:
                    while True:
                        block = handle.read(1024 * 1024)
                        if not block:
                            break
                        digest.update(block)
                return digest.hexdigest()[:16]
            except OSError:
                pass
        try:
            digest.update(inspect.getsource(function).encode("utf-8"))
        except (OSError, TypeError):
            digest.update(repr(function).encode("utf-8"))
        return digest.hexdigest()[:16]

    def _resolve_brics_builder(self):
        try:
            from RL_PPO.moldr import utils as legacy_utils

            BRICSBuild = legacy_utils.BRICSBuild
            fingerprint = self._implementation_fingerprint(BRICSBuild)
            return (
                BRICSBuild,
                "dapigen_custom-sha256-%s" % fingerprint,
                tuple(legacy_utils.reverseReactions),
            )
        except Exception as exc:
            if not self.allow_rdkit_brics_fallback:
                raise RuntimeError(
                    "RL_PPO.moldr.utils.BRICSBuild is required for formal "
                    "Stage-0 runs. The stock RDKit fallback changes the task "
                    "semantics and is permitted only for isolated smoke tests."
                ) from exc
            from rdkit.Chem.BRICS import BRICSBuild

            fingerprint = self._implementation_fingerprint(BRICSBuild)
            return BRICSBuild, "rdkit_fallback-sha256-%s" % fingerprint, tuple()

    def _index_closure_reactions(self, reactions):
        """Index the released reverse BRICS reactions by dummy-atom labels."""

        indexed = defaultdict(list)
        for reaction in reactions:
            matchers = getattr(reaction, "_matchers", ())
            if len(matchers) != 2:
                continue
            labels = []
            for matcher in matchers:
                dummy_atoms = [
                    atom
                    for atom in matcher.GetAtoms()
                    if int(atom.GetAtomicNum()) == 0
                ]
                if len(dummy_atoms) != 1:
                    labels = []
                    break
                labels.append(int(dummy_atoms[0].GetIsotope()))
            if len(labels) == 2:
                indexed[(labels[0], labels[1])].append(reaction)
        return dict((key, tuple(value)) for key, value in indexed.items())

    def _brics_builder(self):
        return self._BRICSBuild, self.assembly_backend_name

    def assemble_candidates(self, base_smiles: str, block_smiles: str) -> Tuple[str, ...]:
        self.diagnostics["assembly_requests"] += 1
        if self.attachment_count(block_smiles) == 0:
            return (self.canonicalize(block_smiles),)
        builder, source = self._brics_builder()
        self.diagnostics["assembly_backend_%s" % source] += 1
        base = self.Chem.MolFromSmiles(base_smiles)
        block = self.Chem.MolFromSmiles(block_smiles)
        if base is None or block is None:
            return tuple()
        generated = builder(
            [base, block],
            onlyCompleteMols=False,
            scrambleReagents=False,
            maxDepth=0,
        )
        candidates = set()
        for molecule in generated:
            try:
                self.Chem.SanitizeMol(molecule)
                candidates.add(self.Chem.MolToSmiles(molecule, canonical=True))
            except Exception:
                self.diagnostics["assembly_invalid_products"] += 1
        self.diagnostics["assembly_products"] += len(candidates)
        return tuple(sorted(candidates))

    def assemble_closure_candidates(
        self, base_smiles: str, block_smiles: str
    ) -> Tuple[str, ...]:
        """Run only label-matched legacy BRICS reactions at final attachment closure.

        The optimized path is valid only when each reactant has exactly one
        attachment. The independent ``assemble_candidates`` implementation is
        retained as the full-enumeration audit reference.
        """

        base = self.Chem.MolFromSmiles(base_smiles)
        block = self.Chem.MolFromSmiles(block_smiles)
        if base is None or block is None:
            return tuple()
        base_dummies = [
            atom for atom in base.GetAtoms() if int(atom.GetAtomicNum()) == 0
        ]
        block_dummies = [
            atom for atom in block.GetAtoms() if int(atom.GetAtomicNum()) == 0
        ]
        if len(base_dummies) != 1 or len(block_dummies) != 1:
            raise ValueError(
                "Closure-specific assembly requires one attachment per reactant."
            )
        if not self._closure_reactions:
            self.diagnostics["closure_assembly_full_fallbacks"] += 1
            return self.assemble_candidates(base_smiles, block_smiles)

        base_label = int(base_dummies[0].GetIsotope())
        block_label = int(block_dummies[0].GetIsotope())
        runs = []
        for reaction in self._closure_reactions.get(
            (base_label, block_label), ()
        ):
            runs.append((reaction, (base, block)))
        for reaction in self._closure_reactions.get(
            (block_label, base_label), ()
        ):
            runs.append((reaction, (block, base)))

        self.diagnostics["closure_assembly_requests"] += 1
        self.diagnostics["closure_assembly_reaction_runs"] += len(runs)
        candidates = set()
        for reaction, reactants in runs:
            try:
                products = reaction.RunReactants(reactants)
            except Exception:
                self.diagnostics["closure_assembly_reaction_failures"] += 1
                continue
            for product_tuple in products:
                if not product_tuple:
                    continue
                molecule = product_tuple[0]
                try:
                    self.Chem.SanitizeMol(molecule)
                    candidates.add(
                        self.Chem.MolToSmiles(molecule, canonical=True)
                    )
                except Exception:
                    self.diagnostics["closure_assembly_invalid_products"] += 1
        self.diagnostics["closure_assembly_products"] += len(candidates)
        return tuple(sorted(candidates))

    def final_polyimide_candidates(
        self, dianhydride_smiles: str, diamine_smiles: str
    ) -> Tuple[str, ...]:
        """Enumerate final products without the released random.choice calls."""

        self.diagnostics["terminal_reaction_requests"] += 1
        dianhydride = self.Chem.MolFromSmiles(dianhydride_smiles)
        diamine = self.Chem.MolFromSmiles(diamine_smiles)
        if dianhydride is None or diamine is None:
            return tuple()
        try:
            first_products = self._first_reaction.RunReactants((dianhydride, diamine))
        except Exception:
            return tuple()

        outputs = set()
        for product_tuple in first_products:
            if not product_tuple:
                continue
            first_product = product_tuple[0]
            try:
                self.Chem.SanitizeMol(first_product)
            except Exception:
                continue

            # The released generate_PI takes the first ReplaceSubstructs result.
            # Keeping this step preserves its reaction semantics while removing
            # downstream stochastic product choice.
            replacements = self.Chem.ReplaceSubstructs(
                first_product,
                self._terminal_amine_pattern,
                self._carbon_replacement,
                replaceAll=False,
            )
            if not replacements:
                continue
            replaced = replacements[0]
            try:
                second_products = self._second_reaction.RunReactants((replaced,))
            except Exception:
                continue
            for second_tuple in second_products:
                if not second_tuple:
                    continue
                candidate = self.Chem.Mol(second_tuple[0])
                for atom in candidate.GetAtoms():
                    if int(atom.GetAtomicNum()) == 0:
                        atom.SetAtomicNum(1)
                try:
                    self.Chem.SanitizeMol(candidate)
                    outputs.add(self.Chem.MolToSmiles(candidate, canonical=True))
                except Exception:
                    self.diagnostics["terminal_invalid_products"] += 1
        self.diagnostics["terminal_products"] += len(outputs)
        return tuple(sorted(outputs))
