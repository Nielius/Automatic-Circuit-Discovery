import logging
import random
from dataclasses import dataclass
import torch.nn.functional as F

import torch
from jaxtyping import Float

from acdc.TLACDCEdge import Edge
from acdc.nudb.adv_opt.data_fetchers import EXPERIMENT_DATA_PROVIDERS, AdvOptExperimentName, AdvOptExperimentData
from acdc.nudb.adv_opt.utils import device

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG)


# Goal: run it once, calculate the metric


# TODO test:
# 1. If you run SP without masking anything, you get the same result as running the entire model
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

    def run_with_individual_metrics(
        self,
        circuit: list[Edge],
        # comparision_circuit: list[Edge] | None = None,
        last_sequence_position_only: bool = False,
    ) -> Float[torch.Tensor, "batch"]:
        """
        Run and calculate an individual metric for each input.

        'last_sequence_position_only' is a flag that should be set to True for tasks where only the last sequence position matters.
        If set to True, the metric will be calculated only for the last sequence position.
        Otherwise, the average metric will be calculated across all sequence positions.

        """
        masked_output_logits: Float[torch.Tensor, "batch pos vocab"] = self.experiment_data.masked_runner.run(
            input=self.experiment_data.task_data.test_data,
            patch_input=self.experiment_data.task_data.test_patch_data,
            edges_to_ablate=list(self.experiment_data.masked_runner.all_ablatable_edges - set(circuit)),
        )
        base_output_logits: Float[torch.Tensor, "batch pos vocab"] = self.experiment_data.masked_runner.run(
            input=self.experiment_data.task_data.test_data,
            patch_input=self.experiment_data.task_data.test_patch_data,
            edges_to_ablate=[],
        )

        # TODO NUDB:
        # depending on the task, we only want to take the last sequence position or not.
        # E.g. for the reverse task, every sequence position matters.
        # But probably for many others it doesn't

        metrics = F.kl_div(
            F.log_softmax(masked_output_logits, dim=-1),
            F.log_softmax(base_output_logits, dim=-1),
            reduction="none",
            log_target=True,
        ).mean(dim=-1)

        if last_sequence_position_only:
            raise NotImplementedError()
        else:
            metrics = metrics.mean(dim=-1)

        return metrics

    def random_circuit(self) -> list[Edge]:
        return [edge for edge in self.experiment_data.masked_runner.all_ablatable_edges if random.random() < 0.4]

    def canonical_circuit_with_random_edges_removed(self, num_of_removals: int) -> list[Edge]:
        return list(
            set(self.experiment_data.circuit_edges)
            - set(random.choices(self.experiment_data.circuit_edges, k=num_of_removals))
        )


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


def main_for_tracr_reverse():
    experiment_tracr_reverse = AdvOptExperiment(
        experiment_data=EXPERIMENT_DATA_PROVIDERS[AdvOptExperimentName.TRACR_REVERSE].get_experiment_data(
            num_examples=6,
            metric_name="l2",
            device=device,
        )
    )

    metrics_with_canonical_circuit = experiment_tracr_reverse.run_with_individual_metrics(
        circuit=experiment_tracr_reverse.experiment_data.circuit_edges
    )
    metrics_with_random_circuit = experiment_tracr_reverse.run_with_individual_metrics(
        circuit=experiment_tracr_reverse.random_circuit()
    )
    metrics_with_corrupted_canonical_circuit = experiment_tracr_reverse.run_with_individual_metrics(
        circuit=experiment_tracr_reverse.canonical_circuit_with_random_edges_removed(2)
    )
    logger.info("Metric is %s", metrics_with_canonical_circuit)

    torch.histogram(metrics_with_canonical_circuit, bins=100)

    # plot histogram of output
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    range = (
        0,
        max(
            metrics_with_canonical_circuit.max().item(),
            metrics_with_random_circuit.max().item(),
            metrics_with_corrupted_canonical_circuit.max().item(),
        ),
    )
    ax.stairs(*torch.histogram(metrics_with_canonical_circuit, bins=100, range=range), label="canonical circuit")
    ax.stairs(*torch.histogram(metrics_with_random_circuit, bins=100, range=range), label="random circuit")
    ax.stairs(
        *torch.histogram(metrics_with_corrupted_canonical_circuit, bins=100, range=range),
        label="corrupted canonical circuit"
    )
    ax.set_xlabel("KL divergence")
    ax.set_ylabel("Frequency")
    ax.set_title("KL divergence for tracr-reverse, histogram")
    ax.legend()

    print(123)


def main_for_tracr_proportion():
    # WARNING: it looks like the ACDC code for proportion is very buggy:
    # - missing BOS
    # - however, in the metrics, it looks like the first row is ignored as if it belongs to the BOS token
    # - tracr's INPUT_ENCODER is not being used properly
    experiment = AdvOptExperiment(
        experiment_data=EXPERIMENT_DATA_PROVIDERS[AdvOptExperimentName.TRACR_PROPORTION].get_experiment_data(
            num_examples=10, metric_name="kl_div", device=device
        )
    )

    metric = experiment.run_my_test()
    logger.info("Metric is %s", metric)

    # Removing two randomly selected edges
    run_with_other_circuit(
        experiment,
        list(
            set(experiment.experiment_data.circuit_edges)
            - set(random.choices(experiment.experiment_data.circuit_edges, k=2))
        ),
    )

    # running the entire model gets you 0.0123
    run_with_other_circuit(
        experiment,
        list(experiment.experiment_data.masked_runner.all_ablatable_edges),
    )

    # which fortunately is the same as what you get when running the entire model, so it still seems to check out
    output_logits = experiment.experiment_data.masked_runner.masked_transformer.model(
        experiment.experiment_data.task_data.test_data
    )
    experiment.experiment_data.task_data.validation_metric(output_logits).item()

    # running the canonical circuit doesn't get 0! it gets 0.0123, same as running the entire model
    run_with_other_circuit(
        experiment,
        experiment.experiment_data.circuit_edges,
    )

    # running with a random circuit gets you up to 0.02, but still relatively low...
    run_with_random_circuit(experiment)


# main_for_tracr_proportion()
main_for_tracr_reverse()
