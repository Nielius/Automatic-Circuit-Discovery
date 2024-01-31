import itertools

import torch
from jaxtyping import Num
from transformer_lens import HookedTransformer

from acdc.TLACDCEdge import HookPointName, IndexedHookPointName, Edge
from subnetwork_probing.sp_utils import MaskedTransformer


class MaskedRunner:
    """
    A class to run a forward pass on a HookedTransformer, with some edges disabled.

    This class is intended to be mostly stateless."""

    masked_transformer: MaskedTransformer

    _parent_index_per_child: dict[tuple[HookPointName, IndexedHookPointName], int]
    _indexed_parents_per_child: dict[HookPointName, list[IndexedHookPointName]]

    def __init__(self, model: HookedTransformer):
        self.masked_transformer = MaskedTransformer(model=model)
        self.masked_transformer.freeze_weights()
        self._freeze_all_masks()
        self._set_all_masks_to_pos_infty()
        self._set_up_parent_index_per_child()

    def _set_up_parent_index_per_child(self):
        # For every child, every possible parent in the edge has an index; this is the index of the mask in the list of masks for that child.
        # This sets up a look-up table for that.
        self._parent_index_per_child = {}
        self._indexed_parents_per_child = {}

        for child, all_parents in self.masked_transformer.parent_node_names.items():
            # we expand the list of all_parents into a list of all indexed parents, so that we can give
            # each of an index
            all_indexed_parents: list[IndexedHookPointName] = list(
                itertools.chain.from_iterable(
                    IndexedHookPointName.list_from_hook_point(name, self.masked_transformer.n_heads)
                    for name in all_parents
                )
            )
            self._indexed_parents_per_child[child] = all_indexed_parents
            for index, indexed_parent in enumerate(all_indexed_parents):
                self._parent_index_per_child[(child, indexed_parent)] = index

    def _freeze_all_masks(self):
        """In the MaskedTransformer, every mask is a parameter. In this case, however,
        we only want to run the model with fixed masks, so we freeze all the masks."""
        for value in self.masked_transformer._mask_logits_dict.values():
            value.requires_grad = False

    def _set_all_masks_to_pos_infty(self):
        for parameter in self.masked_transformer._mask_logits_dict.values():
            parameter.data.fill_(float("inf"))

    def _set_mask_for_edge(self, child: IndexedHookPointName, parent: IndexedHookPointName, value: float) -> None:
        parent_index = self._parent_index_per_child[(child.hook_name, parent)]
        # self._mask_logits_dict is dict[HookPointName of child, Num[torch.nn.Parameter, "parent (IndexedHookPoint), TorchIndex of child"]
        # todo: I think child.index.as_index[-1] shows that we're not using the right abstraction here; or maybe it doesn't?
        self.masked_transformer._mask_logits_dict[child.hook_name][parent_index][child.index.as_index[-1]] = value

    @property
    def all_ablatable_edges(self) -> set[Edge]:
        return {
            Edge(child=indexed_child, parent=indexed_parent)
            for child, all_indexed_parents in self._indexed_parents_per_child.items()
            for indexed_child in IndexedHookPointName.list_from_hook_point(child, self.masked_transformer.n_heads)
            for indexed_parent in all_indexed_parents
        }

    def run(
        self,
        input: Num[torch.Tensor, "batch seq"],
        patch_input: Num[torch.Tensor, "batch seq"],
        edges_to_ablate: list[Edge],
    ) -> Num[torch.Tensor, "batch pos vocab"]:
        for edge in edges_to_ablate:
            assert edge in self.all_ablatable_edges  # safety check
            self._set_mask_for_edge(edge.child, edge.parent, float("-inf"))

        try:
            with self.masked_transformer.with_fwd_hooks_and_new_cache(
                ablation="resample", ablation_data=patch_input
            ) as hooked_model:
                return hooked_model(input)
        finally:
            for edge in edges_to_ablate:
                self._set_mask_for_edge(
                    edge.child, edge.parent, float("inf")
                )  # this class is not intended to keep state
