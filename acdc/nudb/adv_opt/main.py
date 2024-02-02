import logging
import random
from dataclasses import dataclass

import torch
from jaxtyping import Float

from acdc.TLACDCEdge import Edge
from acdc.nudb.adv_opt.data_fetchers import EXPERIMENT_DATA_PROVIDERS, AdvOptExperimentName, AdvOptExperimentData

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG)


# Goal: run it once, calculate the metric


# TODO test:
# 1. If you run the normal mask, you get precisely the same results
# 2.


@dataclass
class AdvOptExperiment:
    experiment_data: AdvOptExperimentData

    def run_my_test(self) -> float:
        output_logits: Float[torch.Tensor, "batch pos vocab"] = self.experiment_data.masked_runner.run(
            input=self.experiment_data.task_data.test_data,
            patch_input=self.experiment_data.task_data.test_patch_data,
            edges_to_ablate=list(self.experiment_data.ablated_edges),
        )
        cur_metric = self.experiment_data.task_data.validation_metric(output_logits).item()

        return cur_metric


experiment = AdvOptExperiment(
    experiment_data=EXPERIMENT_DATA_PROVIDERS[AdvOptExperimentName.TRACR_REVERSE].get_experiment_data()
)

metric = experiment.run_my_test()
logger.info("Metric is %s", metric)


# Ideas:
# 1. run with random circuit
# 2. run with true circuit + some random edges added or removed


def run_with_other_circuit(experiment: AdvOptExperiment, circuit: list[Edge]) -> float:
    output_logits: Float[torch.Tensor, "batch pos vocab"] = experiment.experiment_data.masked_runner.run(
        input=experiment.experiment_data.task_data.test_data,
        patch_input=experiment.experiment_data.task_data.test_patch_data,
        edges_to_ablate=list(experiment.experiment_data.masked_runner.all_ablatable_edges - set(circuit)),
    )
    cur_metric = experiment.experiment_data.task_data.validation_metric(output_logits).item()

    return cur_metric


def run_with_random_circuit(experiment: AdvOptExperiment) -> float:
    return run_with_other_circuit(
        experiment=experiment,
        circuit=[
            edge for edge in experiment.experiment_data.masked_runner.all_ablatable_edges if random.random() < 0.4
        ],
    )

# Removing two randomly selected edges
run_with_other_circuit(experiment, list(set(experiment.experiment_data.circuit_edges) - set(random.choices(experiment.experiment_data.circuit_edges, k=2))))

