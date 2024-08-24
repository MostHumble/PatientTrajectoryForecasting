from dataclasses import dataclass
from typing import Dict, List

from torch import Tensor
from torch.utils.data import DataLoader


def test_get_sequences(pred_trgs : List[List[int]], targets : List[List[int]],
                        source_input_ids : Tensor, test_dataloader : DataLoader,
                        config : dataclass, tgt_tokens_to_ids : Dict[str, int], max_lenght : int):

    for seq in pred_trgs:
        assert len(seq) <= max_lenght + 1, "Generated sequence exceeds maximum length"
        if len(seq) < max_lenght + 1:  # Only check shorter sequences for EOV
            assert seq[-1] == tgt_tokens_to_ids['EOV'] , "EOV token missing in sequence"
    assert len(pred_trgs) == len(targets), "Mismatch in number of predictions and targets"
    assert(len(pred_trgs)) == config.eval_batch_size * (len(test_dataloader)-1) + source_input_ids.size(0), "Not all sequences have been generated"