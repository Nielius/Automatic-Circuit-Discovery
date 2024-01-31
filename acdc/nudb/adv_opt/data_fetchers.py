from acdc.docstring.utils import AllDataThings, get_all_docstring_things, get_docstring_subgraph_true_edges
from acdc.greaterthan.utils import get_all_greaterthan_things, get_greaterthan_true_edges
from acdc.induction.utils import get_all_induction_things
from acdc.nudb.adv_opt.main import true_edges
from acdc.nudb.adv_opt.utils import device, num_examples, metric
from acdc.nudb.joblib_caching import joblib_memory
from acdc.tracr_task.utils import get_all_tracr_things, get_tracr_reverse_edges


@joblib_memory.cache
def get_task_data_tracr_reverse() -> tuple[AllDataThings, dict]:
    # batch size 6, seq len 4
    task_data = get_all_tracr_things(task="reverse", metric_name="l2", num_examples=6, device=device)
    true_edges = get_tracr_reverse_edges()

    return task_data, true_edges


def get_task_data_docstring() -> tuple[AllDataThings, dict]:
    task_data = get_all_docstring_things(num_examples=10, seq_len=4, device=device)
    true_edges = get_docstring_subgraph_true_edges()

    return task_data, true_edges


@joblib_memory.cache
def get_task_data_induction() -> tuple[AllDataThings, dict]:
    # NOTE: this doesn't have a canonical circuit!
    # batch size 6, seq len 4
    task_data = get_all_induction_things(num_examples=10, seq_len=4, device=device)
    raise NotImplementedError("No canoncial circuit for induction yet")

    return task_data, true_edges


@joblib_memory.cache
def get_task_data_greaterthan() -> tuple[AllDataThings, dict]:
    task_data = get_all_greaterthan_things(num_examples=num_examples, metric_name=metric, device=device)
    true_edges = get_greaterthan_true_edges(
        model=task_data.tl_model
    )  # this is now a dict of edges; maybe we want a full TLACDCCorrespondence?

    return task_data, true_edges
