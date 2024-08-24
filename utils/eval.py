from typing import Dict, List

import numpy as np
import torch
from numpy import mean as np_mean
from tqdm import tqdm

from utils.train import create_source_mask, generate_square_subsequent_mask


def get_k(sequence: List[int], k: int, spec_target_ids: torch.Tensor) -> List[int]:
    """
    Returns the first `k` elements from the `sequence` list that are not present in `spec_target_ids`.

    Args:
        sequence (List[int]): The input sequence of integers.
        k (int): The number of elements to return.
        spec_target_ids (torch.Tensor): The tensor containing the specific target IDs.

    Returns:
        List[int]: The first `k` elements from `sequence` that are not present in `spec_target_ids`.
    """
    return torch.tensor(sequence)[~torch.isin(torch.tensor(sequence), spec_target_ids)][
        :k
    ].tolist()


def apk(
    relevant: List[int],
    forecasted: List[int],
    spec_target_ids: torch.Tensor = torch.tensor([0, 1, 2, 3, 4, 5]),
    k: int = 10,
) -> float:
    """
    Calculates the Average Precision at K (AP@K) metric for evaluating the performance of a forecasting model.

    Args:
        relevant (List[int]): The list of relevant items.
        forecasted (List[int]): The list of forecasted items.
        spec_target_ids (torch.Tensor, optional): The specific target IDs to consider. Defaults to torch.tensor([0, 1, 2, 3, 4, 5]).
        k (int, optional): The value of K for calculating AP@K. Defaults to 10.

    Returns:
        float: The Average Precision at K (AP@K) score.

    """
    # filters out special tokens
    forecasted = get_k(forecasted, k, spec_target_ids)
    relevant = get_k(relevant, k, spec_target_ids)
    sum_precision = 0.0
    num_hits = 0.0

    for i, forecast in enumerate(forecasted):

        if forecast in relevant and forecast not in forecasted[:i]:
            num_hits += 1.0
            precision_at_i = num_hits / (i + 1.0)
            sum_precision += precision_at_i

    if num_hits == 0.0:
        return 0.0

    return sum_precision / num_hits


def mapk(relevant: List[List[int]], forecasted: List[List[int]], k: int = 10):
    """
    Calculates the mean average precision at k (MAP@k) for a list of relevant and forecasted items.

    Args:
        relevant (List[List[int]]): A list of lists containing the relevant items for each query.
        forecasted (List[List[int]]): A list of lists containing the forecasted items for each query.
        k (int, optional): The value of k for calculating MAP@k. Defaults to 10.

    Returns:
        float: The mean average precision at k.
    """
    return np_mean([apk(r, f, k=k) for r, f in zip(relevant, forecasted)])


def recallTop(y_true, y_pred, rank=[20, 40, 60]):
    recall = list()
    for i in range(len(y_pred)):
        thisOne = list()
        codes = y_true[i]
        tops = y_pred[i]
        for rk in rank:
            predictions_at_k = len(set(codes).intersection(set(tops[:rk]))) * 1.0
            thisOne.append(predictions_at_k / len(set(codes)))
            recall.append(thisOne)
    return (np.array(recall)).mean(axis=0).tolist()


def get_sequences(
    model,
    dataloader: torch.utils.data.dataloader.DataLoader,
    source_pad_id: int = 0,
    tgt_tokens_to_ids: Dict[str, int] = None,
    max_len: int = 150,
    DEVICE: str = "cuda:0",
):
    """
    return relevant forcasted and sequences made by the model on the dataset.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        val_dataloader (torch.utils.data.DataLoader): The validation dataloader.
        source_pad_id (int, optional): The padding token ID for the source input. Defaults to 0.
        DEVICE (str, optional): The device to run the evaluation on. Defaults to 'cuda:0'.
        tgt_tokens_to_ids (dict, optional): A dictionary mapping target tokens to their IDs. Defaults to None.
        max_len (int, optional): The maximum length of the generated target sequence. Defaults to 100.
    Returns:
        List[List[int]], List[List[int]]: The list of relevant and forecasted sequences.
    """

    model.eval()
    pred_trgs = []
    targets = []
    with torch.inference_mode():
        for source_input_ids, target_input_ids in tqdm(dataloader, desc="scoring"):
            batch_pred_trgs = []
            batch_targets = []
            source_input_ids, target_input_ids = source_input_ids.to(
                DEVICE
            ), target_input_ids.to(DEVICE)
            src_mask, source_padding_mask = create_source_mask(
                source_input_ids, source_pad_id, DEVICE
            )
            memory = model.batch_encode(source_input_ids, src_mask, source_padding_mask)
            pred_trg = (
                torch.tensor(tgt_tokens_to_ids["BOS"], device=DEVICE)
                .repeat(source_input_ids.size(0))
                .unsqueeze(1)
            )
            # generate target sequence one token at a time at batch level
            for i in range(max_len):
                trg_mask = generate_square_subsequent_mask(i + 1, DEVICE)
                output = model.decode(pred_trg, memory, trg_mask)
                probs = model.generator(output[:, -1])
                pred_tokens = torch.argmax(probs, dim=1)
                pred_trg = torch.cat((pred_trg, pred_tokens.unsqueeze(1)), dim=1)
                eov_mask = pred_tokens == tgt_tokens_to_ids["EOV"]

                if eov_mask.any():
                    # extend with sequences that have reached EOV
                    batch_pred_trgs.extend(pred_trg[eov_mask].tolist())
                    batch_targets.extend(target_input_ids[eov_mask].tolist())
                    # break if all have reached EOV
                    if eov_mask.all():
                        break
                    # edit corresponding target sequences
                    target_input_ids = target_input_ids[~eov_mask]
                    pred_trg = pred_trg[~eov_mask]
                    memory = memory[~eov_mask]

            # add elements that have never reached EOV
            if source_input_ids.size(0) != len(batch_pred_trgs):
                batch_pred_trgs.extend(pred_trg.tolist())
                batch_targets.extend(target_input_ids.tolist())
            pred_trgs.extend(batch_pred_trgs)
            targets.extend(batch_targets)
    return pred_trgs, targets


def get_random_stats(
    targets: List[List[int]],
    seq_len: int = 96,
    ks: List[int] = [20, 40, 60],
    num_runs_avg: int = 5,
):
    """
    Returns the average MAP@k and Recall@k scores for a random forecasting model.

    Args:
        targets (List[List[int]]): The list of target sequences.
        seq_len (int, optional): The length of the forecasted sequence. Defaults to 96.
        ks (List[int], optional): The list of k values for MAP@k and Recall@k. Defaults to [20, 40, 60].
        num_runs_avg (int, optional): The number of runs to average the results over. Defaults to 5.
    Returns:
        Dict[str, float], Dict[str, float]: The average MAP@k and Recall@k scores.
    """
    # targets = [concated_dt[i]['target_sequences'].numpy().tolist() for i in range(len(concated_dt))]
    unique_targets = list(set([item for sublist in targets for item in sublist]))

    cumulative_mapk = {f"test_map@{k}": 0.0 for k in ks}
    cumulative_recallk = {f"test_recall@{k}": 0.0 for k in ks}

    for _ in range(num_runs_avg):

        forecasted = [
            np.random.choice(unique_targets, size=seq_len, replace=True).tolist()
            for _ in range(len(targets))
        ]

        run_mapk = {f"test_map@{k}": mapk(targets, forecasted, k) for k in ks}
        run_recallk = {
            f"test_recall@{k}": recallTop(targets, forecasted, rank=[k])[0] for k in ks
        }

        # Accumulate results
        for k in ks:
            cumulative_mapk[f"test_map@{k}"] += run_mapk[f"test_map@{k}"]
            cumulative_recallk[f"test_recall@{k}"] += run_recallk[f"test_recall@{k}"]

    # Compute average results
    average_mapk = {
        f"test_map@{k}": cumulative_mapk[f"test_map@{k}"] / num_runs_avg for k in ks
    }
    average_recallk = {
        f"test_recall@{k}": cumulative_recallk[f"test_recall@{k}"] / num_runs_avg
        for k in ks
    }

    return average_mapk, average_recallk
