import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Optional, List, Dict, Union, Tuple
import random
from dataclasses import asdict
from utils.utils import get_paths, load_data
from model import Seq2SeqTransformer
import numpy as np
import yaml
import os

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

    print(f'Number of unique source codes: {len_unique_source}, max source code: {max_unique_source}')
    print(f'Number of unique target codes: {len_unique_target}, max target code: {max_unique_target}')
    

    embedding_size_source, mapping_source = get_embedding_size(max_unique_source, len_unique_source, unique_source, multiplier)
    embedding_size_target, mapping_target = get_embedding_size(max_unique_target, len_unique_target, unique_target, multiplier)
    
    data_and_properties = {'old_to_new_ids_source' : None, 'old_to_new_ids_target' : None}

    if mapping_source is not None:
        print('reformating source data')
        source_sequences = [[mapping_source[code] for code in sequence] for sequence in source_sequences]
        data_and_properties['old_to_new_ids_source'] = mapping_source
        #mapping_source = {v: k for k, v in mapping_source.items()}

    if mapping_target is not None:
        print('reformatting target data')
        target_sequences = [[mapping_target[code] for code in sequence] for sequence in target_sequences]
        data_and_properties['old_to_new_ids_target'] = mapping_target
        #mapping_target = {v: k for k, v in mapping_target.items()}

    data_and_properties['embedding_size_source'] = embedding_size_source
    data_and_properties['embedding_size_target'] = embedding_size_target
    data_and_properties['source_sequences'] = source_sequences
    data_and_properties['target_sequences'] = target_sequences

    
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

def create_source_mask(src, source_pad_id = 0, DEVICE='cuda:0'):
    """
    Create a mask for the source sequence.

    Args:
        src (torch.Tensor): The source sequence tensor.
        source_pad_id (int, optional): The padding value for the source sequence. Defaults to 0.
        DEVICE (str, optional): The device to be used for computation. Defaults to 'cuda:0'.

    Returns:
        torch.Tensor: The source mask tensor, and the source padding mask.
    """

    src_seq_len = src.shape[1]
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE)
    source_padding_mask = (src == source_pad_id)
    return src_mask, source_padding_mask

def generate_square_subsequent_mask(tgt_seq_len, DEVICE='cuda:0'):
    """
    Generates a square subsequent mask for self-attention mechanism.

    Args:
        sz (int): The size of the mask.
        DEVICE (str, optional): The device to be used for computation. Defaults to 'cuda:0'.

    Returns:
        torch.Tensor: The square subsequent mask.

    """
    mask = (torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_target_mask(tgt, target_pad_id = 0, DEVICE='cuda:0'):
    """
    Create a mask for the target sequence.

    Args:
        tgt (torch.Tensor): The target sequence tensor.
        target_pad_id (int, optional): The padding value for the target sequence. Defaults to 0.
        DEVICE (str, optional): The device to be used for computation. Defaults to 'cuda:0'.

    Returns:
        torch.Tensor: The target mask tensor.
    """

    tgt_seq_len = tgt.shape[1]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, DEVICE)
    tgt_padding_mask = (tgt == target_pad_id)
    return tgt_mask, tgt_padding_mask

def create_mask(src, tgt, source_pad_id = 0, target_pad_id = 0, DEVICE='cuda:0'):
    """
    Create masks for the source and target sequences.

    Args:
        src (torch.Tensor): The source sequence tensor.
        tgt (torch.Tensor): The target sequence tensor.
        source_pad_id (int, optional): The padding value for the source sequence. Defaults to 0.
        target_pad_id (int, optional): The padding value for the target sequence. Defaults to 0.
        DEVICE (str, optional): The device to be used for computation. Defaults to 'cuda:0'.

    Returns:
        torch.Tensor: The source mask tensor.
        torch.Tensor: The target mask tensor.
        torch.Tensor: The source padding mask tensor.
        torch.Tensor: The target padding mask tensor.
    """

    src_mask, source_padding_mask = create_source_mask(src, source_pad_id, DEVICE)
    tgt_mask, target_padding_mask = create_target_mask(tgt, target_pad_id, DEVICE)

    return src_mask, tgt_mask, source_padding_mask, target_padding_mask


def train_epoch(model, optimizer, train_dataloader, loss_fn, source_pad_id = 0, target_pad_id = 0, DEVICE='cuda:0'):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        train_dataloader (DataLoader): The dataloader for training data.
        loss_fn (nn.Module): The loss function used for training.
        source_pad_id (int): The padding token ID for the source input.
        target_pad_id (int): The padding token ID for the target input.
        DEVICE (str, optional): The device to be used for training. Defaults to 'cuda:0'.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    losses = 0

    for source_input_ids, target_input_ids in tqdm(train_dataloader, desc='train'):
        
        source_input_ids = source_input_ids.to(DEVICE)
        target_input_ids = target_input_ids.to(DEVICE)
        target_input_ids_ = target_input_ids[:, :-1]

        source_mask, target_mask, source_padding_mask, target_padding_mask = create_mask(source_input_ids, target_input_ids_, source_pad_id, target_pad_id, DEVICE)
        logits = model(source_input_ids, target_input_ids_, source_mask, target_mask, source_padding_mask, target_padding_mask, source_padding_mask)
    
        _target_input_ids = target_input_ids[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), _target_input_ids.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses += loss.item()

    return losses / len(train_dataloader)

#remember to fix the ids_to_types_map if need to eval based on types of codes (not needed now because only predict diagnosis)

def evaluate(model, val_dataloader, loss_fn,  source_pad_id = 0, target_pad_id = 0, DEVICE='cuda:0'):
    """
    Evaluate the model on the validation dataset.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        val_dataloader (torch.utils.data.DataLoader): The validation dataloader.
        loss_fn (torch.nn.Module): The loss function used for evaluation.
        DEVICE (str, optional): The device to be used for evaluation. Defaults to 'cuda:0'.

    Returns:
        float: The average loss over the validation dataset.
    """
    model.eval()

    losses = 0

    with torch.inference_mode():
        
        for source_input_ids, target_input_ids in tqdm(val_dataloader, desc='evaluation'):
            
            source_input_ids = source_input_ids.to(DEVICE)
            target_input_ids = target_input_ids.to(DEVICE)
            
            target_input_ids_ = target_input_ids[:, :-1]        

            source_mask, target_mask, source_padding_mask, target_padding_mask = create_mask(source_input_ids, target_input_ids_, source_pad_id, target_pad_id, DEVICE)

            logits = model(source_input_ids, target_input_ids_, source_mask, target_mask, source_padding_mask, target_padding_mask, source_padding_mask)

            _target_input_ids = target_input_ids[:, 1:]
            
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), _target_input_ids.reshape(-1))
            losses += loss.item()

    return losses / len(val_dataloader)


def get_data_loaders(train_batch_size=128, eval_batch_size=128, pin_memory=True,
                     seed=213033, test_size=0.05, valid_size=0.05, strategy=None, predict_procedure=None,
                     predict_drugs=None, **kw):
    """
    Get data loaders for training, validation, and testing.

    Args:
        train_batch_size (int): Batch size for training data. Default is 128.
        eval_batch_size (int): Batch size for evaluation data. Default is 128.
        pin_memory (bool): If True, the data loader will pin memory for faster data transfer to GPU. Default is True.
        seed (int): Random seed for data splitting. Default is 89957.
        test_size (float): Fraction of the data to be used for testing. Default is 0.05.
        valid_size (float): Fraction of the data to be used for validation. Default is 0.05.
        strategy (str): Strategy for data loading. Default is None.
        predict_procedure (str): Procedure for prediction. Default is None.
        predict_drugs (str): Drugs for prediction. Default is None.
        **kw: Additional keyword arguments.

    Returns:
        tuple: A tuple containing the train data loader, validation data loader, test data loader,
               source tokens to IDs map, target tokens to IDs map, IDs to types map, and data properties.
    """
    
    with open('PatientTrajectoryForecasting/paths.yaml', 'r') as file:
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

    train_dataloader = DataLoader(train_set, batch_size = train_batch_size, shuffle = True, pin_memory = pin_memory)
    val_dataloader = DataLoader(val_set, batch_size = eval_batch_size, shuffle = False, pin_memory = pin_memory)
    test_dataloader = DataLoader(test_set, batch_size = eval_batch_size, shuffle = False, pin_memory = pin_memory)

    return train_dataloader, val_dataloader, test_dataloader, source_tokens_to_ids, target_tokens_to_ids_map, ids_to_types_map, data_and_properties


def save_checkpoint(epoch, model, optimizer, val_loss = float('inf'), force = False, prefix:str = '', run_num = 0,
                     threshold = 0.01, checkpoint_patience = 8, current_patience = 0, best_val_loss = float('inf')):
    """
    Saves a checkpoint of the model during training if the validation loss improves.

    Args:
        epoch (int): The current epoch number.
        model: The model to be saved.
        optimizer: The optimizer used for training.
        val_loss (float, optional): The current validation loss. Defaults to float('inf').
        force (bool, optional): If True, forces the checkpoint to be saved regardless of the validation loss. Defaults to False.
        prefix (str, optional): A prefix to be added to the checkpoint filename. Defaults to ''.
        run_num (int, optional): The run number of the checkpoint. Defaults to 0.
        threshold (float, optional): The threshold value for improvement in validation loss. Defaults to 0.01.
        checkpoint_patience (int, optional): The number of epochs to wait for improvement in validation loss before stopping training. Defaults to 8.

    Returns:
        bool: False if the checkpoint is not saved, True otherwise.
    """
    model_checkpoint_dir = 'model_checkpoints'
    os.makedirs(model_checkpoint_dir, exist_ok=True)

    if force or (val_loss < (best_val_loss - threshold)):
        best_val_loss = val_loss
        current_patience = 0  # Reset patience counter
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }
        model_checkpoint_path = os.path.join(model_checkpoint_dir, f'{prefix}_best_checkpoint_run_{run_num}.pt')
        torch.save(checkpoint, model_checkpoint_path)
        print(f"Checkpoint saved at {model_checkpoint_path}")
    else:
        current_patience += 1

        if current_patience >= checkpoint_patience:
            print(f"Validation loss hasn't improved for {current_patience} epochs. Stopping training.")
    return False


def load_checkpoint(run = 0, model_checkpoint_dir='/kaggle/working/model_checkpoints',config_dir='/kaggle/working/configs', return_optimizer_state:bool = False, prefix:str = ''):
    
    with open(f'{config_dir}/{prefix}_model_config_run_{run}.yaml', 'r') as yaml_file:
        loaded_model_params = yaml.safe_load(yaml_file)

    # Create a new instance of the model with the loaded configuration
    loaded_transformer = Seq2SeqTransformer(
        loaded_model_params["num_encoder_layers"],
        loaded_model_params["num_decoder_layers"],
        loaded_model_params["emb_size"],
        loaded_model_params["nhead"],
        loaded_model_params["source_vocab_size"],
        loaded_model_params["target_vocab_size"],
        loaded_model_params["ffn_hid_dim"]
    )
    print(loaded_model_params)
    if torch.cuda.is_available():
        checkpoint = torch.load(f'{model_checkpoint_dir}/{prefix}_best_checkpoint_run_{run}.pt')
    else:
        checkpoint = torch.load(f'{model_checkpoint_dir}/{prefix}_best_checkpoint_run_{run}.pt',map_location=torch.device('cpu'))
    
        # Remove the "module." prefix from parameter names caused by trained with ddp
    new_state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove the "module." prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
            
    epoch = checkpoint['epoch']
    loss = checkpoint['best_val_loss']
    loaded_transformer.load_state_dict(new_state_dict)
    if return_optimizer_state:
        return  loaded_transformer, checkpoint['optimizer_state_dict'], epoch, loss
    return loaded_transformer, None, epoch, loss


def save_config(config, run_num= 0,prefix:str =''):
    config_checkpoint_dir = 'configs'
    os.makedirs(config_checkpoint_dir, exist_ok=True)
    # Create a dictionary to store the parameters
    model_params = asdict(config)

    # Save the parameters to a YAML file (e.g., 'model_config.yaml')
    with open(f'{config_checkpoint_dir}/{prefix}_model_config_run_{run_num}.yaml', 'w') as yaml_file:
        yaml.dump(model_params, yaml_file)