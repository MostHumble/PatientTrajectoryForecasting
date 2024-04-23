from torch import triu as torch_triu, ones as torch_ones, bool as torch_bool, zeros as torch_zeros
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Optional, List, Dict, Union, Tuple
import random
from utils.utils import get_paths, load_data
import numpy as np
import yaml

def train_test_val_split(source_sequences : List[List[int]], target_sequences : List[List[int]],
                test_size : float = 0.05, valid_size : float = 0.05, random_state = None)\
    -> Tuple[Tuple[List[int], List[int]], Tuple[List[int], List[int]], Tuple[List[int], List[int]]]:
    """
    Generates the train, test, validation splits for botht source sequences, and target sequences

    Args:
        source_sequences (list): list of source sequences.
        target_sequences (list): list of target sequences.
        test_size (flaot): the fraction of the test set
        valid_size (flaot): the fraction of the test set
        random_state (int): Seed for random number generation.

    Returns:
        Tuple[Tuple[List[int], List[int]], Tuple[List[int], List[int]], Tuple[List[int], List[int]]]:
          A tuple containing the train, test, and validation splits for both source and target sequences.
        
        
    """
    # Validate inputs
    if not 0 <= test_size <= 1 or not 0 <= valid_size <= 1:
        raise ValueError("test_size and valid_size must be in the range [0, 1].")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    source_sequences = np.array(source_sequences)
    target_sequences = np.array(target_sequences)
    dataSize = len(source_sequences)
    idx = np.random.permutation(dataSize)
    nTest = int(np.ceil(test_size * dataSize))
    nValid = int(np.ceil(valid_size * dataSize))
    
    test_idx = idx[:nTest]
    valid_idx = idx[nTest:nTest+nValid]
    train_idx = idx[nTest+nValid:]

    train = {'source_sequences': source_sequences[train_idx], 'target_sequences' : target_sequences[train_idx]}
    test = {'source_sequences': source_sequences[test_idx],  'target_sequences' : target_sequences[test_idx]}
    valid = {'source_sequences':  source_sequences[valid_idx], 'target_sequences' : target_sequences[valid_idx]}
    
    return (train, test, valid)
    

def get_optimal_embedding_size(source_sequences : List[List[int]], target_sequences : List[List[int]], multiplier :int = 64) -> Dict[str, Union[set, int]]:
    """
    Get the unique elements in the given sequences and calculate optimal embedding size, creates corresponding mapping if needed.

    Args:
        source_sequences (list): List of source sequences.
        target_sequences (list): List of target sequences.
        multiplier (int): The multiplier to use for the embedding size.
    Returns:
        Dict[str,Union[int, int, list, list, dict, dict]]: A dictionary containing the optimal embedding size for the source and target sequences, the source and target sequences, and the mapping of the new codes to old codes.
    """
    # Flatten the lists of lists
    source_flat = [item for sublist in source_sequences for item in sublist]
    target_flat = [item for sublist in target_sequences for item in sublist]
    
    # Create sets of unique elements
    unique_source = set(source_flat)
    unique_target = set(target_flat)

    len_unique_source = len(unique_source)
    len_unique_target = len(unique_target)
    max_unique_source = max(unique_source)
    max_unique_target = max(unique_target)
    
    data_and_properties = {}

    embedding_size_source, mapping_source = get_embedding_size(max_unique_source, len_unique_source, unique_source, multiplier)
    embedding_size_target, mapping_target = get_embedding_size(max_unique_target, len_unique_target, unique_target, multiplier)

    if mapping_source is not None:
        source_sequences = [[mapping_source[code] for code in sequence] for sequence in source_sequences]
        data_and_properties['old_to_new_ids_source'] = mapping_source
        #mapping_source = {v: k for k, v in mapping_source.items()}

    if mapping_target is not None:
        target_sequences = [[mapping_target[code] for code in sequence] for sequence in target_sequences]
        data_and_properties['old_to_new_ids_target'] = mapping_target
        #mapping_target = {v: k for k, v in mapping_target.items()}

    data_and_properties['embedding_size_source'] = embedding_size_source
    data_and_properties['embedding_size_target'] = embedding_size_target
    data_and_properties['source_sequences'] = source_sequences
    data_and_properties['target_sequences'] = target_sequences
    data_and_properties['old_to_new_ids_source'] = None
    data_and_properties['old_to_new_ids_target'] = None
    
    return data_and_properties

def get_embedding_size(max_unique : int, len_unique : int, unique_data : set, multiplier :int) -> Tuple[int, Optional[Dict[int, int]]]:
    """
    Get the optimal embedding size for the given data.

    Args:
        max_unique (int): The maximum unique element in the data.
        len_unique (int): The number of unique elements in the data.
        unique_data (set): The set of unique elements in the data.
        multiplier (int): The multiplier to use for the embedding size.
    Returns:
        Tuple[int, Optional[Dict[int, int]]]: A tuple containing the optimal embedding size and a mapping of the old to new codes if the data needs to be reformatted.
    """
    if max_unique - multiplier > len_unique:
        print('must reformat, too much memory would be lost in embedding')
        # create a new mapping of the codes to the new codes
        old_ids_to_new_mapping = {code: i for i, code in enumerate(unique_data)}
        return (len(old_ids_to_new_mapping) // multiplier + 1) * multiplier, old_ids_to_new_mapping
    else:
        # create the embedding size that is rounded to the nearest multiple of 64 of the max unique
        return (max_unique // multiplier + 1) * multiplier, None
    



def enforce_reproducibility(use_seed :Optional[int] = None) -> int:
    """
    enforce reproducibility by setting the seed for all random number generators
    Args:
        use_seed (int): the seed to use, if None, a random seed is generated
    Returns:
        seed (int): the seed used
    """
    seed = use_seed if use_seed is not None else random.randint(1, 1000000)
    print(f"Using seed: {seed}")

    random.seed(seed)    # python RNG
    np.random.seed(seed) # numpy RNG

    # pytorch RNGs
    torch.manual_seed(seed)          # cpu + cuda
    torch.cuda.manual_seed_all(seed) # multi-gpu - can be called without gpus
    if use_seed: # slower speed! https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    return seed


def get_gpu_memory(empty_cache : bool = False):
    """
    Print the amount of free GPU memory.
    Args:
        empty_cache (bool): whether to empty the cache before checking the memory
    """

    if empty_cache:
        torch.cuda.empty_cache()
    mem_alloc = torch.cuda.memory_allocated()
    mem_cached = torch.cuda.memory_reserved()
    mem_free = mem_cached - mem_alloc

    print("Free GPU memory:", mem_free / 1024**3, "GB")

class patientTrajectoryForcastingDataset(Dataset):
    """patientTrajectoryForcastingDataset"""

    def __init__(self, source_sequences, target_sequences, **kw):
        """
        Arguments:
            source_sequences (List[List[int]]]): Path to the csv file with annotations.
        """
        self.source_sequences = source_sequences
        self.target_sequences = target_sequences
        self.len_ = len(self.source_sequences)

    def __len__(self):
        return self.len_

    def __getitem__(self, idx):
        
        return self.source_sequences[idx], self.target_sequences[idx]


def generate_square_subsequent_mask(sz, DEVICE = 'cuda:0'):
    mask = (torch_triu(torch_ones((sz, sz), device = DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, source_pad_id = 0, target_pad_id = 0, DEVICE = 'cuda:0'):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]
    
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch_zeros((src_seq_len, src_seq_len), device = DEVICE).type(torch_bool)

    src_padding_mask = (src == source_pad_id)
    tgt_padding_mask = (tgt == target_pad_id)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def train_epoch(model, optimizer, train_dataloader, loss_fn, DEVICE = 'cuda:0'):
    model.train()
    losses = 0

    for source_input_ids, target_input_ids in tqdm(train_dataloader, desc='train'):
        
        source_input_ids = source_input_ids.to(DEVICE)

        target_input_ids = target_input_ids.to(DEVICE)

        target_input_ids_ = target_input_ids[:, :-1]

        source_mask , target_mask, source_padding_mask, target_padding_mask = create_mask(source_input_ids, target_input_ids_, source_pad_id, target_pad_id)
            
        logits = model(source_input_ids, target_input_ids_, source_mask, target_mask,source_padding_mask, target_padding_mask, source_padding_mask)
    
        optimizer.zero_grad()
        _target_input_ids = target_input_ids[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), _target_input_ids.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))

#remember to fix the ids_to_types_map if need to eval based on types of codes (not needed now because only predict diagnosis)

def evaluate(model, val_dataloader, loss_fn, DEVICE = 'cuda:0'):
    model.eval()
    losses = 0

    for source_input_ids, target_input_ids in tqdm(val_dataloader,desc='evaluation'):
        
        source_input_ids = source_input_ids.to(DEVICE)
        target_input_ids = target_input_ids.to(DEVICE)
        
        target_input_ids_ = target_input_ids[:, :-1]        

        source_mask , target_mask, source_padding_mask, target_padding_mask = create_mask(source_input_ids, target_input_ids_)
        
        logits = model(source_input_ids, target_input_ids_, source_mask, target_mask,source_padding_mask, target_padding_mask, source_padding_mask)

        _target_input_ids = target_input_ids[:, 1:]
        
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), _target_input_ids.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))

def get_data_loaders(train_batch_size = 128, eval_batch_size = 32, num_workers = 5,
                      seed = 0, test_size = 0.05, valid_size = 0.05, strategy = None, predict_procedure = None,\
                      predict_drugs = None, **kw):
    
    with open('paths.yaml', 'r') as file:
        path_config = yaml.safe_load(file)

    train_data_path = get_paths(path_config, strategy, predict_procedure, predict_drugs, train = True, processed_data = True)

    _ , ids_to_types_map, tokens_to_ids_map, __ = load_data(train_data_path['train_data_path'], train = True)

    source_sequences, target_sequences, ___ , ____ = load_data(train_data_path['processed_data_path'], processed_data = True)

    data_and_properties = get_optimal_embedding_size(source_sequences, target_sequences)

    if data_and_properties['old_to_new_ids_target'] is not None:
            target_tokens_to_ids_map = {token: data_and_properties['old_to_new_ids_target'][idx] for token, idx in tokens_to_ids_map.items() if idx in data_and_properties['old_to_new_ids_target'].keys()}
    else:
        target_tokens_to_ids_map = tokens_to_ids_map

    if data_and_properties['old_to_new_ids_source'] is not None:
        source_tokens_to_ids = {token: data_and_properties['old_to_new_ids_source'][idx] for token, idx in tokens_to_ids_map.items() if idx in data_and_properties['old_to_new_ids_source'].keys()}   
    else:
        source_tokens_to_ids = tokens_to_ids_map

    train, test, val = train_test_val_split(data_and_properties['source_sequences'], data_and_properties['target_sequences'], test_size = test_size, valid_size = valid_size, random_state = seed)

    train_set  = patientTrajectoryForcastingDataset(**train)
    test_set  = patientTrajectoryForcastingDataset(**test)
    val_set  = patientTrajectoryForcastingDataset(**val)

    train_dataloader = DataLoader(train_set, batch_size = train_batch_size, shuffle = True,  num_workers = num_workers)
    val_dataloader = DataLoader(val_set, batch_size = eval_batch_size, shuffle = False,  num_workers = num_workers)
    test_dataloader = DataLoader(test_set, batch_size = eval_batch_size, shuffle = False,  num_workers = num_workers)

    return train_dataloader, val_dataloader, test_dataloader, source_tokens_to_ids, target_tokens_to_ids_map, ids_to_types_map, data_and_properties
