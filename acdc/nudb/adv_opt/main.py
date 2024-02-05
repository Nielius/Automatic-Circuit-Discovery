import datetime
import itertools
import logging
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from jaxtyping import Float

from acdc.TLACDCEdge import Edge
from acdc.nudb.adv_opt.data_fetchers import EXPERIMENT_DATA_PROVIDERS, AdvOptExperimentName, AdvOptExperimentData
from acdc.nudb.adv_opt.utils import device, CIRCUITBENCHMARKS_DATA_DIR

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
    ) -> Float[torch.Tensor, "batch"]:
        """
        Run and calculate an individual metric for each input. The metric compares the output for the circuit
        with the output for the full model.

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

        if self.experiment_data.metric_last_sequence_position_only:
            # depending on the task, we only want to take the last sequence position or not.
            # E.g. for the reverse task, every sequence position matters.
            # But for e.g. the docstring task, we only want to get the metrics
            # from the final sequence position.
            metrics = F.kl_div(
                F.log_softmax(masked_output_logits[:, -1, :], dim=-1),
                F.log_softmax(base_output_logits[:, -1, :], dim=-1),
                reduction="none",
                log_target=True,
            ).mean(dim=-1)
        else:
            metrics = (
                F.kl_div(
                    F.log_softmax(masked_output_logits, dim=-1),
                    F.log_softmax(base_output_logits, dim=-1),
                    reduction="none",
                    log_target=True,
                )
                .mean(dim=-1)
                .mean(dim=-1)
            )  # mean over sequence position and output logit

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


def main_for_plotting_three_experiments(experiment_name: AdvOptExperimentName):
    logger.info("Starting plotting experiment for '%s'.", experiment_name)
    experiment = AdvOptExperiment(
        experiment_data=EXPERIMENT_DATA_PROVIDERS[experiment_name].get_experiment_data(
            num_examples=6,
            metric_name="l2" if experiment_name == AdvOptExperimentName.TRACR_REVERSE else "kl_div",
            device=device,
        )
    )

    logger.info("Running with all edges")
    metrics_with_full_model = experiment.run_with_individual_metrics(
        circuit=list(experiment.experiment_data.masked_runner.all_ablatable_edges)
    )

    logger.info("Running with canonical circuit")
    metrics_with_canonical_circuit = experiment.run_with_individual_metrics(
        circuit=experiment.experiment_data.circuit_edges
    )

    logger.info("Running with a random circuit")
    metrics_with_random_circuit = experiment.run_with_individual_metrics(circuit=experiment.random_circuit())

    logger.info("Running with the canonical circuit, but with 2 random edges removed")
    metrics_with_corrupted_canonical_circuit = experiment.run_with_individual_metrics(
        circuit=experiment.canonical_circuit_with_random_edges_removed(2)
    )

    def plot():
        # plot histogram of output
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)  # Create a figure containing a single axes.
        ((ax_all, ax_1), (ax_2, ax_3)) = axes
        range = (
            0,
            max(
                metrics_with_full_model.max().item(),
                metrics_with_canonical_circuit.max().item(),
                metrics_with_random_circuit.max().item(),
                metrics_with_corrupted_canonical_circuit.max().item(),
            ),
        )
        ax_all.stairs(*torch.histogram(metrics_with_full_model, bins=100, range=range), label="full model")
        ax_all.stairs(
            *torch.histogram(metrics_with_canonical_circuit, bins=100, range=range), label="canonical circuit"
        )
        ax_all.stairs(*torch.histogram(metrics_with_random_circuit, bins=100, range=range), label="random circuit")
        ax_all.stairs(
            *torch.histogram(metrics_with_corrupted_canonical_circuit, bins=100, range=range),
            label="corrupted canonical circuit",
        )
        fig.suptitle(
            f"KL divergence between output of the full model and output of a circuit, for {experiment_name}, histogram"
        )
        ax_1.stairs(*torch.histogram(metrics_with_canonical_circuit, bins=100, range=range), label="canonical circuit")
        ax_2.stairs(*torch.histogram(metrics_with_random_circuit, bins=100, range=range), label="random circuit")
        ax_3.stairs(
            *torch.histogram(metrics_with_corrupted_canonical_circuit, bins=100, range=range),
            label="corrupted canonical circuit",
        )
        for ax in itertools.chain(*axes):
            ax.set_xlabel("KL divergence")
            ax.set_ylabel("Frequency")
            ax.legend()

        plot_dir = CIRCUITBENCHMARKS_DATA_DIR / "plots"
        plot_dir.mkdir(exist_ok=True)
        figure_path = plot_dir / f"{experiment_name}_histogram_{datetime.datetime.now().isoformat()}.png"
        fig.savefig(figure_path)
        logger.info("Saved histogram to %s", figure_path)

    plot()

    topk_most_adversarial = torch.topk(metrics_with_random_circuit, k=5, sorted=True)
    topk_most_adversarial_input = experiment.experiment_data.task_data.test_data[topk_most_adversarial.indices, :]

    if experiment.experiment_data.masked_runner.masked_transformer.model.tokenizer is not None:
        # decode if a tokenizer is given
        topk_most_adversarial_input = [
            experiment.experiment_data.masked_runner.masked_transformer.model.tokenizer.decode(input)
            for input in topk_most_adversarial_input
        ]

    logger.info(
        "Top 5 most adversarial examples with loss '%s': %s",
        topk_most_adversarial.values,
        topk_most_adversarial_input,
    )


    # Debugging code: decode the input data with tokenizer
    # experiment.experiment_data.masked_runner.masked_transformer.model.tokenizer.decode(
    #     experiment.experiment_data.task_data.test_data[0]
    # )


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


if __name__ == "__main__":
    logger.info("Using device %s", device)

    # main_for_tracr_proportion()
    main_for_plotting_three_experiments(AdvOptExperimentName.TRACR_REVERSE)
    # main_for_plotting_three_experiments(AdvOptExperimentName.DOCSTRING)
    # main_for_plotting_three_experiments(AdvOptExperimentName.GREATERTHAN)
    # main_for_plotting_three_experiments(AdvOptExperimentName.IOI)
