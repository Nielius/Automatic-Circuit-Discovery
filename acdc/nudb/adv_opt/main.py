import logging
import random
from dataclasses import dataclass
from typing import Callable

import torch
from jaxtyping import Shaped, Float, Num
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCEdge import EdgeWithInfo, Edge, IndexedHookPointName
from acdc.acdc_graphics import graph_from_edges
from acdc.nudb.adv_opt.data_fetchers import get_task_data_tracr_reverse
from acdc.nudb.adv_opt.masked_runner import MaskedRunner

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG)

# true edges is
# dict[tuple[TargetHookName, TargetHookindex, SourceHookName, SourceHookIndex], bool]
task_data, true_edges = get_task_data_tracr_reverse()


# Goal: run it once, calculate the metric


@dataclass
class CompleteMask:
    # little bit like the TLACDCCorrespondence
    values: dict[IndexedHookPointName, dict[IndexedHookPointName, float]]

    @classmethod
    def initialize_from_model(cls, model: HookedTransformer) -> "CompleteMask":
        """Create a CompleteMask with all the maskable values set to 1.0.
        The main point of this function is that it gives you a list of all
        the edges that you can mask."""
        pass

    @classmethod
    def _get_maskable_edges_from_model(cls, model: HookedTransformer) -> list[Edge]:
        pass


# TODO test:
# 1. If you run the normal mask, you get precisely the same results
# 2.

@dataclass
class AdvOptExperiment:
    runner: MaskedRunner
    metric: Callable[[torch.Tensor], torch.Tensor]
    dataset: Shaped[
        torch.Tensor, "batch *data"
    ]  # for greater than, it's int64; also see https://docs.kidger.site/jaxtyping/api/array/
    corrupted_dataset: Shaped[torch.Tensor, "batch *data"]

    def run_my_test(self) -> float:
        logits: Float[torch.Tensor, "batch pos vocab"] = self.runner.run(
            input=self.dataset,
            patch_input=self.corrupted_dataset,
            edges_to_ablate=[
                edge
                for edge in self.runner.all_ablatable_edges
                if random.random() < 0.5
                # Edge(
                #     child=IndexedHookPointName(
                #         hook_name="blocks.3.hook_resid_post",
                #         index=TorchIndex([None]),
                #     ),
                #     parent=IndexedHookPointName(
                #         hook_name="blocks.2.hook_mlp_out",
                #         index=TorchIndex([None]),
                #     ),
                # )
            ],
        )
        cur_metric = self.metric(logits).item()

        return cur_metric


experiment = AdvOptExperiment(
    runner=MaskedRunner(model=task_data.tl_model),
    metric=task_data.validation_metric,
    dataset=task_data.test_data,
    corrupted_dataset=task_data.test_patch_data,
)

metric = experiment.run_my_test()
logger.info("Metric is %s", metric)
