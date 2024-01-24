from acdc.greaterthan.utils import get_greaterthan_true_edges, get_all_greaterthan_things

num_examples = 5
metric = "kl_div"
device = "cpu"

task_data = get_all_greaterthan_things(num_examples=num_examples, metric_name=metric, device=device)

get_greaterthan_true_edges(model=task_data.tl_model)  # this is now a dict of edges; maybe we want a full TLACDCCorrespondence?