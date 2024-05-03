import torch
from tqdm import tqdm
from utils.train import create_source_mask, generate_square_subsequent_mask
from typing import List, Dict
from numpy import mean as np_mean

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
    return torch.tensor(sequence)[~torch.isin(torch.tensor(sequence), spec_target_ids)][:k].tolist()

def apk(relevant: List[int], forecasted: List[int], spec_target_ids: torch.Tensor = torch.tensor([0, 1, 2, 3, 4, 5]), k: int = 10) -> float:
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
    forecasted = get_k(forecasted, k, spec_target_ids)
    relevant = get_k(relevant, k, spec_target_ids)
    sum_precision = 0.0
    num_hits = 0.0

    for i, forecast in enumerate(forecasted):
        if forecast in relevant and forecast not in forecasted[:i]:
            num_hits += 1.0
            precision_at_i = num_hits / (i + 1.0)
            sum_precision += precision_at_i

    return sum_precision / (num_hits + 1e-10)

def mapk(relevant: List[List[int]], forecasted: List[List[int]], k :int = 10):
    """
    Calculates the mean average precision at k (MAP@k) for a list of relevant and forecasted items.

    Args:
        relevant (List[List[int]]): A list of lists containing the relevant items for each query.
        forecasted (List[List[int]]): A list of lists containing the forecasted items for each query.
        k (int, optional): The value of k for calculating MAP@k. Defaults to 10.

    Returns:
        float: The mean average precision at k.
    """
    return np_mean([apk(r, f, k = k) for r, f in zip(relevant, forecasted)])
    
def get_sequences(model, dataloader : torch.utils.data.dataloader.DataLoader,  source_pad_id : int = 0, tgt_tokens_to_ids : Dict[str, int] =  None, max_len : int = 150,  DEVICE : str ='cuda:0'):
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
        for source_input_ids, target_input_ids in tqdm(dataloader, desc='scoring'):
            source_input_ids, target_input_ids = source_input_ids.to(DEVICE),target_input_ids.to(DEVICE)
            src_mask, source_padding_mask = create_source_mask(source_input_ids, source_pad_id, DEVICE) 
            memory = model.batch_encode(source_input_ids, src_mask, source_padding_mask)
            pred_trg = torch.tensor(tgt_tokens_to_ids['BOS'], device= DEVICE).repeat(source_input_ids.size(0)).unsqueeze(1)
            # generate target sequence one token at a time at batch level
            for i in range(max_len):
                trg_mask = generate_square_subsequent_mask(i+1, DEVICE)
                output = model.decode(pred_trg, memory, trg_mask)
                probs = model.generator(output[:, -1])
                pred_tokens = torch.argmax(probs, dim=1)
                eov_mask = pred_tokens == tgt_tokens_to_ids['EOV']
                if eov_mask.any():
                    # extend with sequences that have reached EOV
                    pred_trgs.extend(torch.cat((pred_trg[eov_mask],torch.tensor(tgt_tokens_to_ids['EOV'], device = DEVICE).unsqueeze(0).repeat(eov_mask.sum(), 1)),dim = -1).cpu().tolist())
                    targets.extend(target_input_ids[eov_mask].cpu().tolist())
                    # store corresponding target sequences
                    target_input_ids = target_input_ids[~eov_mask]
                    # break if all have reached EOV
                    if eov_mask.all():
                        break  
                    pred_trg = torch.cat((pred_trg[~eov_mask], pred_tokens[~eov_mask].unsqueeze(1)), dim=1)
                    memory = memory[~eov_mask]
                else:
                    pred_trg = torch.cat((pred_trg, pred_tokens.unsqueeze(1)), dim=1)
                
    return pred_trgs, targets
