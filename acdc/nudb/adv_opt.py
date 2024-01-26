from dataclasses import dataclass
from typing import Callable

import torch
from jaxtyping import Shaped, Float

from acdc.docstring.utils import AllDataThings
from acdc.greaterthan.utils import get_greaterthan_true_edges, get_all_greaterthan_things
from acdc.nudb.joblib_caching import joblib_memory
from subnetwork_probing.transformer_lens.transformer_lens import HookedTransformer

num_examples = 5
metric = "kl_div"
device = "cpu"

@joblib_memory.cache
def get_task_data() -> tuple[AllDataThings, dict]:
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


@dataclass
class AdvOptExperiment:
    runner: AblationRunner
    metric: Callable[[torch.Tensor], torch.Tensor]
    dataset: Shaped[torch.Tensor, "batch *data"]  # for greater than, it's int64; also see https://docs.kidger.site/jaxtyping/api/array/

    def run(self) -> float:
        logits: Float[torch.Tensor, "batch pos vocab"] = self.runner.model(self.dataset)
        cur_metric = self.metric(logits).item()

        return cur_metric


experiment = AdvOptExperiment(
    runner=AblationRunner(
        model=task_data.tl_model
    ),
    metric=task_data.validation_metric,
    dataset=task_data.test_data,
)

experiment.run()



