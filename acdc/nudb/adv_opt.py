from dataclasses import dataclass
from typing import Callable

import torch
from jaxtyping import Shaped, Float, Num
from torch._C import _get_cublas_allow_fp16_reduced_precision_reduction
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.acdc_graphics import graph_from_edges
from acdc.docstring.utils import AllDataThings
from acdc.greaterthan.utils import get_greaterthan_true_edges, get_all_greaterthan_things
from acdc.nudb.joblib_caching import joblib_memory
from acdc.tracr_task.utils import get_all_tracr_things, get_tracr_reverse_edges

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG)

num_examples = 5
metric = "kl_div"
device = "cpu"


@joblib_memory.cache
def get_task_data() -> tuple[AllDataThings, dict]:
    # batch size 6, seq len 4
    task_data = get_all_tracr_things(task="reverse", metric_name="l2", num_examples=6, device=device)
    true_edges = get_tracr_reverse_edges()

    return task_data, true_edges


# true edges is
# dict[tuple[TargetHookName, TargetHookindex, SourceHookName, SourceHookIndex], bool]


@joblib_memory.cache
def get_task_data_greaterthan() -> tuple[AllDataThings, dict]:
    task_data = get_all_greaterthan_things(num_examples=num_examples, metric_name=metric, device=device)
    true_edges = get_greaterthan_true_edges(
        model=task_data.tl_model
    )  # this is now a dict of edges; maybe we want a full TLACDCCorrespondence?

    return task_data, true_edges


task_data, true_edges = get_task_data()


# Goal: run it once, calculate the metric


@dataclass
class AblationRunner:
    model: HookedTransformer

    def run_with_ablated_edges(
        self,
        edges_to_ablate: list,
        input: Num[torch.Tensor, "batch seq"],
        corrupted_input: Num[torch.Tensor, "batch seq"],
    ):
        # batch size 6, input data dim 4
        output_original, _ = self.model.run_with_cache(input)
        output_on_corrupted_input, corrupted_values_cache = self.model.run_with_cache(corrupted_input)

        base_correspondence = TLACDCCorrespondence.setup_from_model(
            self.model, True if self.model.cfg.positional_embedding_type == "standard" else NotImplementedError()
        )

        all_edges_collection = base_correspondence.edge_dict()
        g = graph_from_edges(edge_collection=all_edges_collection, filename="nielstest.png", show_everything=True)

        def example_hook(hook_point_out: torch.Tensor, hook: HookPoint) -> torch.Tensor:
            print(123)
            return hook_point_out + 1000

        hook_name = "blocks.3.hook_resid_post"

        output = self.model.run_with_hooks(input, fwd_hooks=[(hook_name, example_hook)])

        print(123)
        pass


@dataclass
class AdvOptExperiment:
    runner: AblationRunner
    metric: Callable[[torch.Tensor], torch.Tensor]
    dataset: Shaped[
        torch.Tensor, "batch *data"
    ]  # for greater than, it's int64; also see https://docs.kidger.site/jaxtyping/api/array/
    corrupted_dataset: Shaped[torch.Tensor, "batch *data"]

    def run_my_test(self) -> float:
        logits: Float[torch.Tensor, "batch pos vocab"] = self.runner.run_with_ablated_edges(
            [], input=self.dataset, corrupted_input=self.corrupted_dataset
        )
        cur_metric = self.metric(logits).item()

        return cur_metric

    def run_plain(self) -> float:
        logits: Float[torch.Tensor, "batch pos vocab"] = self.runner.model(self.dataset)
        cur_metric = self.metric(logits).item()

        return cur_metric


experiment = AdvOptExperiment(
    runner=AblationRunner(model=task_data.tl_model),
    metric=task_data.validation_metric,
    dataset=task_data.test_data,
    corrupted_dataset=task_data.test_patch_data,
)

print("wtf")
metric = experiment.run_my_test()
logger.info("Metric is %s", metric)
