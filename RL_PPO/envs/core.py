from __future__ import annotations

import hashlib
import json
from collections import OrderedDict, defaultdict
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import DAPiGenEnvConfig
from .rng import named_index
from .types import (
    ActionMask,
    CoreTransition,
    DAPiGenAction,
    DAPiGenState,
    STATE_SCHEMA_VERSION,
    TERMINATION_ATOM_LIMIT,
    TERMINATION_HORIZON,
    TERMINATION_INVALID_ACTION,
    TERMINATION_NO_CONTINUATION,
    TERMINATION_NO_PRODUCT,
    TERMINATION_PI_REACTION_FAILED,
    TERMINATION_SUCCESS,
    TRANSITION_SNAPSHOT_SCHEMA_VERSION,
)


class BranchableDAPiGenCore(object):
    """Side-effect-free DAPiGen state machine shared by all Stage-0 methods.

    Invariant: for identical ``state``, ``action`` and ``seed``, ``transition``
    returns the same result. The core never invokes a property predictor and
    therefore cannot leak terminal reward into product selection or partial
    transitions.
    """

    def __init__(
        self,
        dianhydride_blocks: Sequence[str],
        diamine_blocks: Sequence[str],
        initial_dianhydride_smiles: str,
        initial_diamine_smiles: str,
        chemistry,
        encoder,
        config: Optional[DAPiGenEnvConfig] = None,
        maximum_cache_entries: int = 200000,
    ) -> None:
        self.config = config or DAPiGenEnvConfig()
        self.chemistry = chemistry
        self.encoder = encoder
        self.maximum_cache_entries = int(maximum_cache_entries)
        if self.maximum_cache_entries <= 0:
            raise ValueError("maximum_cache_entries must be positive.")

        self.initial_dianhydride_smiles = chemistry.canonicalize(
            initial_dianhydride_smiles
        )
        self.initial_diamine_smiles = chemistry.canonicalize(initial_diamine_smiles)
        self.dianhydride_blocks = tuple(
            chemistry.canonicalize(value) for value in dianhydride_blocks
        )
        self.diamine_blocks = tuple(
            chemistry.canonicalize(value) for value in diamine_blocks
        )
        if len(set(self.dianhydride_blocks)) != len(self.dianhydride_blocks):
            raise ValueError(
                "The dianhydride catalog contains canonical duplicate actions. "
                "Deduplicate it in the factory before constructing the core."
            )
        if len(set(self.diamine_blocks)) != len(self.diamine_blocks):
            raise ValueError(
                "The diamine catalog contains canonical duplicate actions. "
                "Deduplicate it in the factory before constructing the core."
            )

        self.dianhydride_metadata = tuple(
            chemistry.block_metadata(index, smiles, "dianhydride")
            for index, smiles in enumerate(self.dianhydride_blocks)
        )
        self.diamine_metadata = tuple(
            chemistry.block_metadata(index, smiles, "diamine")
            for index, smiles in enumerate(self.diamine_blocks)
        )
        self.dianhydride_noop_id = len(self.dianhydride_blocks)
        self.diamine_noop_id = len(self.diamine_blocks)

        self._candidate_cache = OrderedDict()
        self._closure_candidate_cache = OrderedDict()
        self._mask_cache = OrderedDict()
        self._observation_cache = OrderedDict()
        self._embedding_dim = None
        self._diagnostics = defaultdict(int)
        self.environment_id = self._build_environment_id()

        initial = self.initial(seed=0)
        if initial.terminated or initial.truncated:
            raise ValueError("The configured initial state has no valid continuation.")

    @staticmethod
    def _catalog_hash(values: Sequence[str]) -> str:
        payload = "\n".join(values).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _build_environment_id(self) -> str:
        payload = {
            "config": self.config.to_dict(),
            "state_schema_version": STATE_SCHEMA_VERSION,
            "chemistry_backend": getattr(
                self.chemistry, "backend_version", type(self.chemistry).__name__
            ),
            "encoder_version": getattr(
                self.encoder, "encoder_version", type(self.encoder).__name__
            ),
            "dianhydride_catalog": self._catalog_hash(self.dianhydride_blocks),
            "diamine_catalog": self._catalog_hash(self.diamine_blocks),
            "initial_dianhydride": self.initial_dianhydride_smiles,
            "initial_diamine": self.initial_diamine_smiles,
        }
        serialized = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return "dapigen:" + hashlib.sha256(serialized).hexdigest()[:24]

    def _assert_state_compatible(self, state: DAPiGenState) -> None:
        if not isinstance(state, DAPiGenState):
            raise TypeError("state must be a DAPiGenState instance.")
        if state.environment_id != self.environment_id:
            raise ValueError(
                "State environment_id %s does not match core %s."
                % (state.environment_id, self.environment_id)
            )
        if int(state.max_steps) != int(self.config.max_steps):
            raise ValueError("State max_steps does not match the core configuration.")

    def specification(self) -> Dict[str, Any]:
        initial = self.initial(seed=0)
        product_estimand = (
            "common-uniform-quantile-over-sorted-unique-products-v1"
            if self.config.product_selection == "seeded_uniform"
            else "lexicographically-first-enumerated-product-v1"
        )
        return {
            "environment_id": self.environment_id,
            "environment_version": self.config.environment_version,
            "state_schema_version": STATE_SCHEMA_VERSION,
            "config": self.config.to_dict(),
            "product_selection_estimand": product_estimand,
            "observation_dimension": int(initial.observation.size),
            "number_of_dianhydride_actions": len(self.dianhydride_blocks) + 1,
            "number_of_diamine_actions": len(self.diamine_blocks) + 1,
            "dianhydride_noop_id": self.dianhydride_noop_id,
            "diamine_noop_id": self.diamine_noop_id,
            "dianhydride_catalog_sha256": self._catalog_hash(
                self.dianhydride_blocks
            ),
            "diamine_catalog_sha256": self._catalog_hash(self.diamine_blocks),
            "initial_dianhydride_smiles": self.initial_dianhydride_smiles,
            "initial_diamine_smiles": self.initial_diamine_smiles,
            "chemistry_backend": getattr(
                self.chemistry, "backend_version", type(self.chemistry).__name__
            ),
            "encoder_version": getattr(
                self.encoder, "encoder_version", type(self.encoder).__name__
            ),
        }

    def diagnostics(self) -> Dict[str, Any]:
        payload = dict((key, int(value)) for key, value in self._diagnostics.items())
        payload.update(
            {
                "candidate_cache_entries": len(self._candidate_cache),
                "closure_candidate_cache_entries": len(
                    self._closure_candidate_cache
                ),
                "mask_cache_entries": len(self._mask_cache),
                "observation_cache_entries": len(self._observation_cache),
            }
        )
        if hasattr(self.encoder, "diagnostics"):
            payload["encoder"] = self.encoder.diagnostics()
        if hasattr(self.chemistry, "diagnostics"):
            payload["chemistry"] = dict(self.chemistry.diagnostics)
        return payload

    def _cache_get(self, cache: OrderedDict, key: Any):
        if key not in cache:
            return None
        value = cache.pop(key)
        cache[key] = value
        return value

    def _cache_put(self, cache: OrderedDict, key: Any, value: Any) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.maximum_cache_entries:
            cache.popitem(last=False)

    def _normalize_attachment_labels(self, smiles: str) -> str:
        normalized = smiles
        for label in range(1, int(self.config.attachment_label_max) + 1):
            normalized = normalized.replace(
                "([%d*])" % label, "([*])"
            ).replace("[%d*]" % label, "[*]")
        return normalized

    def _terminal_mask(self) -> ActionMask:
        d_mask = np.zeros(len(self.dianhydride_blocks) + 1, dtype=np.bool_)
        a_mask = np.zeros(len(self.diamine_blocks) + 1, dtype=np.bool_)
        d_mask[self.dianhydride_noop_id] = True
        a_mask[self.diamine_noop_id] = True
        return ActionMask(d_mask, a_mask)

    def _make_state(
        self,
        dianhydride_smiles: str,
        diamine_smiles: str,
        dianhydride_complete: bool,
        diamine_complete: bool,
        dianhydride_growth_steps: int,
        diamine_growth_steps: int,
        step_index: int,
        terminated: bool = False,
        truncated: bool = False,
        termination_reason: Optional[str] = None,
    ) -> DAPiGenState:
        return DAPiGenState(
            dianhydride_smiles=self.chemistry.canonicalize(dianhydride_smiles),
            diamine_smiles=self.chemistry.canonicalize(diamine_smiles),
            environment_id=self.environment_id,
            dianhydride_complete=bool(dianhydride_complete),
            diamine_complete=bool(diamine_complete),
            dianhydride_growth_steps=int(dianhydride_growth_steps),
            diamine_growth_steps=int(diamine_growth_steps),
            step_index=int(step_index),
            max_steps=int(self.config.max_steps),
            terminated=bool(terminated),
            truncated=bool(truncated),
            termination_reason=termination_reason,
        )

    def initial(self, seed: int = 0) -> CoreTransition:
        del seed
        state = self._make_state(
            self.initial_dianhydride_smiles,
            self.initial_diamine_smiles,
            False,
            False,
            0,
            0,
            0,
        )
        mask, has_continuation = self._valid_action_mask_and_status(state)
        if not has_continuation:
            state = self._make_state(
                state.dianhydride_smiles,
                state.diamine_smiles,
                False,
                False,
                0,
                0,
                0,
                terminated=True,
                termination_reason=TERMINATION_NO_CONTINUATION,
            )
            mask = self._terminal_mask()
        return CoreTransition(
            state=state,
            observation=self.observe(state),
            action_mask=mask,
            terminated=state.terminated,
            truncated=state.truncated,
            info={
                "environment_id": self.environment_id,
                "state_id": state.state_id,
                "transition_seed": None,
            },
        )

    def observe(self, state: DAPiGenState) -> np.ndarray:
        self._assert_state_compatible(state)
        cached = self._cache_get(self._observation_cache, state)
        if cached is not None:
            self._diagnostics["observation_cache_hits"] += 1
            return cached.copy()
        self._diagnostics["observation_cache_misses"] += 1
        d_smiles = self._normalize_attachment_labels(state.dianhydride_smiles)
        a_smiles = self._normalize_attachment_labels(state.diamine_smiles)
        d_vector = np.asarray(self.encoder(d_smiles), dtype=np.float32).reshape(-1)
        a_vector = np.asarray(self.encoder(a_smiles), dtype=np.float32).reshape(-1)
        if d_vector.size != a_vector.size:
            raise ValueError("The two molecular embeddings have different dimensions.")
        if self._embedding_dim is None:
            self._embedding_dim = int(d_vector.size)
        elif int(d_vector.size) != self._embedding_dim:
            raise ValueError("The molecular encoder dimension changed during execution.")

        if self.config.observation_mode == "legacy_polybert":
            observation = np.concatenate([d_vector, a_vector]).astype(np.float32)
        else:
            max_atoms = float(max(int(self.config.max_atoms), 1))
            max_steps = float(max(int(self.config.max_steps), 1))
            if self.config.observation_mode == "markov_v1":
                metadata = np.asarray(
                    [
                        float(state.dianhydride_complete),
                        float(state.diamine_complete),
                        float(state.step_index) / max_steps,
                        float(state.remaining_steps) / max_steps,
                        float(self.chemistry.atom_count(state.dianhydride_smiles))
                        / max_atoms,
                        float(self.chemistry.atom_count(state.diamine_smiles))
                        / max_atoms,
                        float(
                            self.chemistry.attachment_count(
                                state.dianhydride_smiles
                            )
                        )
                        / 4.0,
                        float(self.chemistry.attachment_count(state.diamine_smiles))
                        / 4.0,
                    ],
                    dtype=np.float32,
                )
            else:
                metadata = np.asarray(
                    [
                        float(state.dianhydride_complete),
                        float(state.diamine_complete),
                        float(state.dianhydride_growth_steps == 0),
                        float(state.diamine_growth_steps == 0),
                        float(state.step_index) / max_steps,
                        float(state.remaining_steps) / max_steps,
                        float(state.dianhydride_growth_steps) / max_steps,
                        float(state.diamine_growth_steps) / max_steps,
                        float(self.chemistry.atom_count(state.dianhydride_smiles))
                        / max_atoms,
                        float(self.chemistry.atom_count(state.diamine_smiles))
                        / max_atoms,
                        float(
                            self.chemistry.attachment_count(
                                state.dianhydride_smiles
                            )
                        )
                        / 4.0,
                        float(self.chemistry.attachment_count(state.diamine_smiles))
                        / 4.0,
                    ],
                    dtype=np.float32,
                )
                if self.config.observation_mode == "augmented_v3":
                    if not hasattr(self.chemistry, "attachment_label_histogram"):
                        raise TypeError(
                            "augmented_v3 requires chemistry.attachment_label_histogram()."
                        )
                    label_max = int(self.config.attachment_label_max)
                    d_labels = np.asarray(
                        self.chemistry.attachment_label_histogram(
                            state.dianhydride_smiles, label_max
                        ),
                        dtype=np.float32,
                    ).reshape(-1)
                    a_labels = np.asarray(
                        self.chemistry.attachment_label_histogram(
                            state.diamine_smiles, label_max
                        ),
                        dtype=np.float32,
                    ).reshape(-1)
                    expected = label_max + 1
                    if d_labels.size != expected or a_labels.size != expected:
                        raise ValueError(
                            "Attachment-label histograms must contain labels 0..%d."
                            % label_max
                        )
                    metadata = np.concatenate(
                        [metadata, d_labels / 4.0, a_labels / 4.0]
                    ).astype(np.float32)
            observation = np.concatenate([d_vector, a_vector, metadata]).astype(
                np.float32
            )
        self._cache_put(self._observation_cache, state, observation.copy())
        return observation

    def _candidate_key(self, side: str, state: DAPiGenState, action_id: int):
        if side == "dianhydride":
            base = state.dianhydride_smiles
            growth = state.dianhydride_growth_steps
        else:
            base = state.diamine_smiles
            growth = state.diamine_growth_steps
        return side, base, int(growth), int(action_id)

    def _minimum_growth(self, side: str) -> int:
        if side == "dianhydride":
            return int(self.config.minimum_dianhydride_growth_steps)
        return int(self.config.minimum_diamine_growth_steps)

    def _current_growth(self, side: str, state: DAPiGenState) -> int:
        if side == "dianhydride":
            return int(state.dianhydride_growth_steps)
        return int(state.diamine_growth_steps)

    def _complete_block_allowed(
        self, side: str, state: DAPiGenState, canonical_block: str
    ) -> bool:
        del canonical_block
        policy = self.config.complete_block_policy
        current_growth = self._current_growth(side, state)
        proposed_growth = current_growth + 1
        if proposed_growth < self._minimum_growth(side):
            return False
        if policy == "never":
            return False
        if policy == "legacy_replace":
            return True
        return current_growth == 0

    def _viable_candidates(
        self, side: str, state: DAPiGenState, action_id: int
    ) -> Tuple[str, ...]:
        key = self._candidate_key(side, state, action_id)
        cached = self._cache_get(self._candidate_cache, key)
        if cached is not None:
            self._diagnostics["candidate_cache_hits"] += 1
            return cached
        self._diagnostics["candidate_cache_misses"] += 1

        if side == "dianhydride":
            metadata = self.dianhydride_metadata[action_id]
            base_smiles = state.dianhydride_smiles
            is_complete = self.chemistry.is_complete_dianhydride
        elif side == "diamine":
            metadata = self.diamine_metadata[action_id]
            base_smiles = state.diamine_smiles
            is_complete = self.chemistry.is_complete_diamine
        else:
            raise ValueError("Unknown side: %s" % side)

        if metadata.is_complete_for_side:
            if self._complete_block_allowed(side, state, metadata.canonical_smiles):
                result = (metadata.canonical_smiles,)
            else:
                result = tuple()
            self._cache_put(self._candidate_cache, key, result)
            return result
        if not metadata.has_attachment:
            self._cache_put(self._candidate_cache, key, tuple())
            return tuple()

        raw_candidates = self.chemistry.assemble_candidates(
            base_smiles, metadata.canonical_smiles
        )
        canonical = []
        for smiles in raw_candidates:
            try:
                canonical.append(self.chemistry.canonicalize(smiles))
            except Exception:
                self._diagnostics["candidate_canonicalization_failures"] += 1

        proposed_growth = self._current_growth(side, state) + 1
        complete = sorted(set(smiles for smiles in canonical if is_complete(smiles)))
        completion_allowed = proposed_growth >= self._minimum_growth(side)
        if complete and completion_allowed:
            result = tuple(complete)
        else:
            # In a controlled long-horizon variant, premature complete products
            # are discarded and any still-open products remain available.
            partial = sorted(
                set(
                    smiles
                    for smiles in canonical
                    if self.chemistry.attachment_count(smiles) > 0
                )
            )
            result = tuple(partial)
        self._cache_put(self._candidate_cache, key, result)
        return result

    def _closure_action_viable(
        self, side: str, state: DAPiGenState, action_id: int
    ) -> bool:
        """Check a one-attachment closure without full BRICS enumeration."""

        key = self._candidate_key(side, state, action_id)
        cached = self._cache_get(self._closure_candidate_cache, key)
        if cached is not None:
            self._diagnostics["closure_candidate_cache_hits"] += 1
            return bool(cached)
        self._diagnostics["closure_candidate_cache_misses"] += 1

        if side == "dianhydride":
            metadata = self.dianhydride_metadata[action_id]
            base_smiles = state.dianhydride_smiles
            is_complete = self.chemistry.is_complete_dianhydride
        elif side == "diamine":
            metadata = self.diamine_metadata[action_id]
            base_smiles = state.diamine_smiles
            is_complete = self.chemistry.is_complete_diamine
        else:
            raise ValueError("Unknown side: %s" % side)

        proposed_growth = self._current_growth(side, state) + 1
        completion_allowed = proposed_growth >= self._minimum_growth(side)
        if not completion_allowed:
            viable = False
        elif hasattr(self.chemistry, "assemble_closure_candidates"):
            candidates = self.chemistry.assemble_closure_candidates(
                base_smiles, metadata.canonical_smiles
            )
            viable = any(is_complete(smiles) for smiles in candidates)
        else:
            # Test doubles and third-party backends retain exact semantics.
            viable = bool(self._viable_candidates(side, state, action_id))
        self._cache_put(self._closure_candidate_cache, key, bool(viable))
        return bool(viable)

    def _side_action_valid(
        self,
        side: str,
        state: DAPiGenState,
        action_id: int,
        base_attachment_labels=None,
        mask_mode: Optional[str] = None,
    ) -> bool:
        if side == "dianhydride":
            completed = state.dianhydride_complete
            noop_id = self.dianhydride_noop_id
            metadata = self.dianhydride_metadata
            base = state.dianhydride_smiles
        else:
            completed = state.diamine_complete
            noop_id = self.diamine_noop_id
            metadata = self.diamine_metadata
            base = state.diamine_smiles
        if completed:
            return int(action_id) == int(noop_id)
        if int(action_id) == int(noop_id):
            return False
        if int(action_id) < 0 or int(action_id) >= len(metadata):
            return False
        block = metadata[int(action_id)]
        if block.is_complete_for_side:
            return self._complete_block_allowed(side, state, block.canonical_smiles)
        if not block.has_attachment:
            return False
        selected_mask_mode = self.config.mask_mode if mask_mode is None else mask_mode
        if selected_mask_mode not in (
            "closure_exact_cached",
            "compatibility",
            "exact_cached",
            "all",
        ):
            raise ValueError("Unsupported mask mode: %s" % selected_mask_mode)
        if selected_mask_mode == "all":
            return True
        if selected_mask_mode == "exact_cached":
            # This path is the independent reference used to audit whether the
            # cheap label rule has false negatives. Do not prefilter it through
            # the compatibility rule being audited.
            return bool(self._viable_candidates(side, state, int(action_id)))
        if base_attachment_labels is None:
            base_attachment_labels = self.chemistry.attachment_labels(base)
        if hasattr(self.chemistry, "labels_can_react"):
            compatible = self.chemistry.labels_can_react(
                base_attachment_labels, block.attachment_labels
            )
        else:
            compatible = self.chemistry.can_react(base, block.canonical_smiles)
        if not compatible:
            return False
        if selected_mask_mode == "closure_exact_cached":
            # A BRICS connection consumes one attachment from each reactant.
            # Label compatibility is usually sufficient while an attachment
            # remains, because any reaction product is still a viable partial
            # structure. At the closure boundary, however, the product is only
            # viable if it is a complete side-specific monomer. Validate just
            # those actions with exact chemistry and cache the result.
            remaining_attachments = (
                int(self.chemistry.attachment_count(base))
                + int(block.attachment_count)
                - 2
            )
            if remaining_attachments <= 0:
                self._diagnostics["closure_exact_checks"] += 1
                viable = self._closure_action_viable(
                    side, state, int(action_id)
                )
                self._diagnostics[
                    "closure_exact_valid" if viable else "closure_exact_rejected"
                ] += 1
                return viable
        return True

    def raw_action_mask_for_mode(
        self, state: DAPiGenState, mask_mode: str
    ) -> Mapping[str, np.ndarray]:
        """Build a side-wise raw mask without dead-end terminal fallback."""

        self._assert_state_compatible(state)
        if state.done:
            terminal = self._terminal_mask()
            return {
                "dianhydride": terminal.dianhydride.copy(),
                "diamine": terminal.diamine.copy(),
            }
        if mask_mode not in (
            "closure_exact_cached",
            "compatibility",
            "exact_cached",
            "all",
        ):
            raise ValueError("Unsupported mask mode: %s" % mask_mode)
        d_mask = np.zeros(len(self.dianhydride_blocks) + 1, dtype=np.bool_)
        a_mask = np.zeros(len(self.diamine_blocks) + 1, dtype=np.bool_)
        if state.dianhydride_complete:
            d_mask[self.dianhydride_noop_id] = True
        else:
            labels = self.chemistry.attachment_labels(state.dianhydride_smiles)
            for action_id in range(len(self.dianhydride_blocks)):
                d_mask[action_id] = self._side_action_valid(
                    "dianhydride",
                    state,
                    action_id,
                    labels,
                    mask_mode=mask_mode,
                )
        if state.diamine_complete:
            a_mask[self.diamine_noop_id] = True
        else:
            labels = self.chemistry.attachment_labels(state.diamine_smiles)
            for action_id in range(len(self.diamine_blocks)):
                a_mask[action_id] = self._side_action_valid(
                    "diamine",
                    state,
                    action_id,
                    labels,
                    mask_mode=mask_mode,
                )
        # Do not wrap these arrays in ActionMask: an audit/reference mode may
        # legitimately expose an all-False side before the formal environment
        # converts a dead end to its terminal NOOP mask.
        return {"dianhydride": d_mask, "diamine": a_mask}

    def compare_compatibility_and_exact_masks(
        self, state: DAPiGenState
    ) -> Mapping[str, Mapping[str, np.ndarray]]:
        """Build label-only, closure-refined and exact masks for an audit."""

        self._assert_state_compatible(state)
        if state.done:
            terminal = self._terminal_mask()
            return {
                "compatibility": {
                    "dianhydride": terminal.dianhydride.copy(),
                    "diamine": terminal.diamine.copy(),
                },
                "refined": {
                    "dianhydride": terminal.dianhydride.copy(),
                    "diamine": terminal.diamine.copy(),
                },
                "exact": {
                    "dianhydride": terminal.dianhydride.copy(),
                    "diamine": terminal.diamine.copy(),
                },
            }

        masks = {}
        for mode in (
            "compatibility",
            "closure_exact_cached",
            "exact_cached",
        ):
            mask = self.raw_action_mask_for_mode(state, mode)
            masks[mode] = {
                "dianhydride": mask["dianhydride"].copy(),
                "diamine": mask["diamine"].copy(),
            }
        return {
            "compatibility": masks["compatibility"],
            "refined": masks["closure_exact_cached"],
            "exact": masks["exact_cached"],
        }

    def _valid_action_mask_and_status(
        self, state: DAPiGenState
    ) -> Tuple[ActionMask, bool]:
        self._assert_state_compatible(state)
        if state.done:
            return self._terminal_mask(), False
        cached = self._cache_get(self._mask_cache, state)
        if cached is not None:
            self._diagnostics["mask_cache_hits"] += 1
            mask, status = cached
            return mask.copy(), bool(status)
        self._diagnostics["mask_cache_misses"] += 1

        raw_mask = self.raw_action_mask_for_mode(state, self.config.mask_mode)
        d_mask = raw_mask["dianhydride"]
        a_mask = raw_mask["diamine"]
        d_ok = bool(
            state.dianhydride_complete or d_mask[:-1].any()
        )
        a_ok = bool(state.diamine_complete or a_mask[:-1].any())
        has_continuation = bool(d_ok and a_ok)
        if not has_continuation:
            d_mask[:] = False
            a_mask[:] = False
            d_mask[self.dianhydride_noop_id] = True
            a_mask[self.diamine_noop_id] = True
        result = ActionMask(d_mask, a_mask)
        self._cache_put(self._mask_cache, state, (result.copy(), has_continuation))
        return result, has_continuation

    def valid_action_mask(self, state: DAPiGenState) -> ActionMask:
        mask, _ = self._valid_action_mask_and_status(state)
        return mask

    def _choose(
        self, candidates: Sequence[str], seed: int, stream: str
    ) -> Tuple[str, int]:
        ordered = tuple(sorted(candidates))
        if not ordered:
            raise ValueError("Cannot choose from an empty candidate set.")
        if self.config.product_selection == "canonical_first":
            index = 0
        else:
            index = named_index(seed, stream, len(ordered))
        return ordered[index], int(index)

    def _failure_transition(
        self,
        state: DAPiGenState,
        reason: str,
        action: DAPiGenAction,
        seed: int,
        details: Optional[Mapping[str, Any]] = None,
        proposed_dianhydride: Optional[str] = None,
        proposed_diamine: Optional[str] = None,
        proposed_dianhydride_complete: Optional[bool] = None,
        proposed_diamine_complete: Optional[bool] = None,
        proposed_dianhydride_growth_steps: Optional[int] = None,
        proposed_diamine_growth_steps: Optional[int] = None,
    ) -> CoreTransition:
        next_state = self._make_state(
            proposed_dianhydride or state.dianhydride_smiles,
            proposed_diamine or state.diamine_smiles,
            (
                state.dianhydride_complete
                if proposed_dianhydride_complete is None
                else proposed_dianhydride_complete
            ),
            (
                state.diamine_complete
                if proposed_diamine_complete is None
                else proposed_diamine_complete
            ),
            (
                state.dianhydride_growth_steps
                if proposed_dianhydride_growth_steps is None
                else proposed_dianhydride_growth_steps
            ),
            (
                state.diamine_growth_steps
                if proposed_diamine_growth_steps is None
                else proposed_diamine_growth_steps
            ),
            min(int(state.step_index) + 1, int(state.max_steps)),
            terminated=True,
            termination_reason=reason,
        )
        info = {
            "environment_id": self.environment_id,
            "previous_state_id": state.state_id,
            "previous_state_snapshot": state.to_dict(),
            "state_id": next_state.state_id,
            "transition_seed": int(seed),
            "action": action.as_tuple(),
            "termination_reason": reason,
        }
        if details:
            info.update(dict(details))
        return CoreTransition(
            state=next_state,
            observation=self.observe(next_state),
            action_mask=self._terminal_mask(),
            terminated=True,
            truncated=False,
            info=info,
        )

    def restore_transition(self, snapshot: Mapping[str, Any]) -> CoreTransition:
        """Reconstruct a transition snapshot without invoking the evaluator.

        This is used for exact process/checkpoint restoration. Successful terminal
        snapshots retain their selected terminal molecule and revalidate the
        chemistry-derived candidate set, unlike legacy state-only restoration.
        """

        snapshot_schema = snapshot.get("snapshot_schema_version")
        if snapshot_schema != TRANSITION_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported or unsafe transition snapshot schema: %s"
                % snapshot_schema
            )
        snapshot_environment = snapshot.get("environment_id")
        if snapshot_environment != self.environment_id:
            raise ValueError(
                "Snapshot environment_id %s does not match core %s."
                % (snapshot_environment, self.environment_id)
            )
        state = DAPiGenState.from_dict(snapshot["state"])
        self._assert_state_compatible(state)
        if (
            self.chemistry.canonicalize(state.dianhydride_smiles)
            != state.dianhydride_smiles
        ):
            raise ValueError("Snapshot dianhydride SMILES is not canonical.")
        if (
            self.chemistry.canonicalize(state.diamine_smiles)
            != state.diamine_smiles
        ):
            raise ValueError("Snapshot diamine SMILES is not canonical.")
        if bool(
            self.chemistry.is_complete_dianhydride(state.dianhydride_smiles)
        ) != bool(state.dianhydride_complete):
            raise ValueError(
                "Snapshot dianhydride completion flag is inconsistent."
            )
        if bool(
            self.chemistry.is_complete_diamine(state.diamine_smiles)
        ) != bool(state.diamine_complete):
            raise ValueError(
                "Snapshot diamine completion flag is inconsistent."
            )
        terminal_smiles = snapshot.get("terminal_smiles")
        terminal_candidates = tuple(snapshot.get("terminal_candidates", ()))
        if terminal_smiles is not None:
            canonical_terminal = self.chemistry.canonicalize(terminal_smiles)
            if canonical_terminal != terminal_smiles:
                raise ValueError("Snapshot terminal_smiles is not canonical.")
        canonical_candidates = tuple(
            sorted(
                set(
                    self.chemistry.canonicalize(item)
                    for item in terminal_candidates
                )
            )
        )
        if canonical_candidates != terminal_candidates:
            raise ValueError(
                "Snapshot terminal_candidates must be canonical, sorted and unique."
            )
        if state.termination_reason == TERMINATION_SUCCESS:
            expected_candidates = tuple(
                sorted(
                    set(
                        self.chemistry.canonicalize(item)
                        for item in self.chemistry.final_polyimide_candidates(
                            state.dianhydride_smiles,
                            state.diamine_smiles,
                        )
                    )
                )
            )
            if terminal_candidates != expected_candidates:
                raise ValueError(
                    "Snapshot terminal_candidates do not match the terminal "
                    "chemistry result."
                )
        return CoreTransition(
            state=state,
            observation=self.observe(state),
            action_mask=self.valid_action_mask(state),
            terminated=state.terminated,
            truncated=state.truncated,
            terminal_smiles=terminal_smiles,
            terminal_candidates=terminal_candidates,
            info={
                "environment_id": self.environment_id,
                "state_id": state.state_id,
                "restored": True,
            },
        )

    def transition(
        self, state: DAPiGenState, action: DAPiGenAction, seed: int
    ) -> CoreTransition:
        self._assert_state_compatible(state)
        if state.done:
            raise ValueError("Cannot transition from an already finished state.")
        action = DAPiGenAction.from_any(action)
        mask = self.valid_action_mask(state)
        d_id, a_id = action.as_tuple()
        d_valid = 0 <= d_id < mask.dianhydride.size and bool(mask.dianhydride[d_id])
        a_valid = 0 <= a_id < mask.diamine.size and bool(mask.diamine[a_id])
        if not (d_valid and a_valid):
            if self.config.invalid_action_handling == "raise":
                raise ValueError(
                    "Action %r is invalid at state %s." % (action, state.state_id)
                )
            return self._failure_transition(
                state,
                TERMINATION_INVALID_ACTION,
                action,
                seed,
                {"dianhydride_valid": d_valid, "diamine_valid": a_valid},
            )

        candidate_counts = {}
        selected_indices = {}
        next_d = state.dianhydride_smiles
        next_a = state.diamine_smiles
        next_d_growth = int(state.dianhydride_growth_steps)
        next_a_growth = int(state.diamine_growth_steps)

        for side, action_id in (("dianhydride", d_id), ("diamine", a_id)):
            if side == "dianhydride":
                completed = state.dianhydride_complete
                noop_id = self.dianhydride_noop_id
                base = state.dianhydride_smiles
            else:
                completed = state.diamine_complete
                noop_id = self.diamine_noop_id
                base = state.diamine_smiles
            if completed:
                if int(action_id) != int(noop_id):
                    return self._failure_transition(
                        state, TERMINATION_INVALID_ACTION, action, seed
                    )
                chosen = base
                candidate_counts[side] = 1
                selected_indices[side] = 0
            else:
                candidates = self._viable_candidates(side, state, int(action_id))
                if not candidates:
                    return self._failure_transition(
                        state,
                        TERMINATION_NO_PRODUCT,
                        action,
                        seed,
                        {"failed_side": side, "failed_action_id": int(action_id)},
                    )
                chosen, chosen_index = self._choose(
                    candidates, seed, "%s_product" % side
                )
                candidate_counts[side] = len(candidates)
                selected_indices[side] = chosen_index
                if side == "dianhydride":
                    next_d_growth += 1
                else:
                    next_a_growth += 1
            if side == "dianhydride":
                next_d = chosen
            else:
                next_a = chosen

        d_complete = bool(
            state.dianhydride_complete
            or self.chemistry.is_complete_dianhydride(next_d)
        )
        a_complete = bool(
            state.diamine_complete or self.chemistry.is_complete_diamine(next_a)
        )
        next_step = int(state.step_index) + 1
        atom_count = int(self.chemistry.atom_count(next_d)) + int(
            self.chemistry.atom_count(next_a)
        )
        base_info = {
            "environment_id": self.environment_id,
            "previous_state_id": state.state_id,
            "previous_state_snapshot": state.to_dict(),
            "transition_seed": int(seed),
            "action": action.as_tuple(),
            "candidate_counts": candidate_counts,
            "selected_candidate_indices": selected_indices,
            "dianhydride_growth_steps": next_d_growth,
            "diamine_growth_steps": next_a_growth,
            "total_monomer_atom_count": atom_count,
        }

        if atom_count > int(self.config.max_atoms):
            return self._failure_transition(
                state,
                TERMINATION_ATOM_LIMIT,
                action,
                seed,
                base_info,
                proposed_dianhydride=next_d,
                proposed_diamine=next_a,
                proposed_dianhydride_complete=d_complete,
                proposed_diamine_complete=a_complete,
                proposed_dianhydride_growth_steps=next_d_growth,
                proposed_diamine_growth_steps=next_a_growth,
            )

        terminal_smiles = None
        terminal_candidates = tuple()
        terminated = False
        truncated = False
        reason = None
        if d_complete and a_complete:
            terminal_candidates = tuple(
                sorted(
                    set(
                        self.chemistry.canonicalize(smiles)
                        for smiles in self.chemistry.final_polyimide_candidates(
                            next_d, next_a
                        )
                    )
                )
            )
            if not terminal_candidates:
                reason = TERMINATION_PI_REACTION_FAILED
                terminated = True
            else:
                terminal_smiles, terminal_index = self._choose(
                    terminal_candidates, seed, "terminal_polyimide_product"
                )
                base_info["terminal_candidate_count"] = len(terminal_candidates)
                base_info["selected_terminal_candidate_index"] = terminal_index
                reason = TERMINATION_SUCCESS
                terminated = True
        elif next_step >= int(self.config.max_steps):
            reason = TERMINATION_HORIZON
            if self.config.horizon_semantics == "failure_terminal":
                terminated = True
            else:
                truncated = True

        next_state = self._make_state(
            next_d,
            next_a,
            d_complete,
            a_complete,
            next_d_growth,
            next_a_growth,
            next_step,
            terminated=terminated,
            truncated=truncated,
            termination_reason=reason,
        )

        if not (terminated or truncated):
            next_mask, has_continuation = self._valid_action_mask_and_status(next_state)
            if not has_continuation:
                next_state = self._make_state(
                    next_d,
                    next_a,
                    d_complete,
                    a_complete,
                    next_d_growth,
                    next_a_growth,
                    next_step,
                    terminated=True,
                    termination_reason=TERMINATION_NO_CONTINUATION,
                )
                terminated = True
                reason = TERMINATION_NO_CONTINUATION
                next_mask = self._terminal_mask()
        else:
            next_mask = self._terminal_mask()

        base_info["state_id"] = next_state.state_id
        base_info["state_snapshot"] = next_state.to_dict()
        base_info["termination_reason"] = reason
        if self.config.include_terminal_candidates_in_info:
            base_info["terminal_candidates"] = terminal_candidates
        return CoreTransition(
            state=next_state,
            observation=self.observe(next_state),
            action_mask=next_mask,
            terminated=terminated,
            truncated=truncated,
            terminal_smiles=terminal_smiles,
            terminal_candidates=terminal_candidates,
            info=base_info,
        )
