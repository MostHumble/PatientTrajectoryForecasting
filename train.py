from utils.data_processing import format_data, prepare_sequences, filter_codes
from utils.utils import load_data, get_paths, store_files, enforce_reproducibility
from dataclasses import dataclass,asdict
import yaml
import torch
import math
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR
from tqdm import tqdm 
from utils.train import train_epoch, evaluate, patientTrajectoryForcastingDataset, train_test_val_split
import wandb
import argparse
import os
from model import Seq2SeqTransformer



from dataclasses import dataclass

@dataclass
class Config:
    seed : int = None
    strategy = 'SDP'
    predict_procedure : bool = False
    predict_drugs : bool = False
    procedure : bool = not(predict_procedure)
    drugs : bool = not(predict_drugs)
    truncate : bool = True
    pad : bool = True
    input_max_length :int = 448
    target_max_length :int = 64
    test_size : float = 0.05
    valid_size : float = 0.05
    source_vocab_size : int = None
    target_vocab_size : int = None
    num_encoder_layers: int = 5
    num_decoder_layers: int = 5
    nhead: int = 2
    emb_size: int = 512
    ffn_hid_dim: int = 2048
    train_batch_size: int = 64
    eval_batch_size: int = 256
    learning_rate: float = 3e-4
    warmup_start: float = 5
    num_train_epochs: int = 25
    warmup_epochs: int = None
    label_smoothing : float = 0.0

def train_transformer(config, train_dataloader, val_dataloader):
    print('Training the model')

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transformer = Seq2SeqTransformer(config.num_encoder_layers, config.num_decoder_layers, config.emb_size,
                                 config.nhead, config.source_vocab_size,
                                 config.target_vocab_size, config.ffn_hid_dim)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        transformer = nn.DataParallel(transformer)

    transformer = transformer.to(DEVICE)
    # remember to fix target_pad_id
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index= config.target_pad_id, label_smoothing = config.label_smoothing)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2), eps=config.eps)

    
    for epoch in range(config.num_train_epochs):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, train_dataloader, loss_fn, DEVICE)
        end_time = timer()
        print(f"Epoch {epoch} Training Loss {train_loss} Time {end_time - start_time}")
        if epoch % 5 == 0:
            evaluate(transformer, val_dataloader, loss_fn, DEVICE)
    



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--wandb', type = bool, default = False, help = 'Whether to use wandb or not')
    parser.add_argument('--wandb_project', type = str, default = 'transformer', help = 'The name of the wandb project')
    parser.add_argument('--wandb_run_name', type = str, default = 'vanilla Transformer', help = 'The name of the wandb run')
    parser.add_argument('--wandb_entity', type = str, default = None, help = 'The name of the wandb entity')
    parser.add_argument('--wandb_api_key', type = str, default = None, help = 'The wandb api key')
    parser.add_argument('--wandb_run_id', type = str, default = None, help = 'The wandb run id')
    parser.add_argument('--wandb_resume', type = bool, default = False, help = 'Whether to resume a wandb run or not')
    parser.add_argument('--wandb_run_notes', type = str, default = None, help = 'The notes of the wandb run')
    parser.add_argument('--wandb_tags', type = list, default = None, help = 'The tags of the wandb run')
    parser.add_argument('--wandb_config', type = dict, default = None, help = 'The config of the wandb run')
    parser.add_argument('--wandb_dir', type = str, default = None, help = 'The directory of the wandb run')
    parser.add_argument('--wandb_mode', type = str, default = 'disabled', help = 'The mode of the wandb run')
    parser.add_argument('--wandb_save_model', type = bool, default = False, help = 'Whether to save the model to wandb or not')
    parser.add_argument('--wandb_save_model_name', type = str, default = 'model', help = 'The name of the model to save to wandb')
    parser.add_argument('--wandb_save_model_notes', type = str, default = None, help = 'The notes of the model to save to wandb')
    parser.add_argument('--wandb_save_model_tags', type = list, default = None, help = 'The tags of the model to save to wandb')
    parser.add_argument('--wandb_save_model_dir', type = str, default = None, help = 'The directory of the model to save to wandb')
    parser.add_argument('--wandb_save_model_policy', type = str, default = 'live', help = 'The policy of the model to save to wandb')
    parser.add_argument('--wandb_save_model_framework', type = str, default = 'pytorch', help = 'The framework of the model to save to wandb')
    parser.add_argument('--wandb_save_model_config', type = dict, default = None, help = 'The config of the model to save to wandb')
    parser.add_argument('--wandb_save_model_files', type = list, default = None, help = 'The files of the model to save to wandb')
    parser.add_argument('--wandb_save_model_id', type = str, default = None, help = 'The id of the model to save to wandb')
    parser.add_argument('--learning_rate', type = float, default = 3e-4, help = 'The learning rate of the model')
    # adam parameters
    parser.add_argument('--beta1', type = float, default = 0.9, help = 'The beta1 parameter of the Adam optimizer')
    parser.add_argument('--beta2', type = float, default = 0.999, help = 'The beta2 parameter of the Adam optimizer')
    parser.add_argument('--eps', type = float, default = 1e-8, help = 'The eps parameter of the Adam optimizer')
    # learning rate scheduler
    parser.add_argument('--lr_scheduler', type = str, default = 'ReduceLROnPlateau', help = 'The learning rate scheduler to use')
    parser.add_argument('--lr_scheduler_factor', type = float, default = 0.1, help = 'The factor parameter of the learning rate scheduler')
    parser.add_argument('--lr_scheduler_patience', type = int, default = 10, help = 'The patience parameter of the learning rate scheduler')
    parser.add_argument('--lr_scheduler_threshold', type = float, default = 1e-4, help = 'The threshold parameter of the learning rate scheduler')
    parser.add_argument('--lr_scheduler_threshold_mode', type = str, default = 'rel', help = 'The threshold mode parameter of the learning rate scheduler')

    parser.add_argument('--num_train_epochs', type = int, default = 25, help = 'The number of training epochs')
    parser.add_argument('--train_batch_size', type = int, default = 64, help = 'The batch size of the training data')
    # arguments for the transformer model
    parser.add_argument('--num_encoder_layers', type = int, default = 5, help = 'The number of encoder layers')
    parser.add_argument('--num_decoder_layers', type = int, default = 5, help = 'The number of decoder layers')
    parser.add_argument('--nhead', type = int, default = 2, help = 'The number of heads in the multiheadattention models')
    parser.add_argument('--emb_size', type = int, default = 512, help = 'The embedding size of the model')
    parser.add_argument('--ffn_hid_dim', type = int, default = 2048, help = 'The feedforward network hidden dimension')
    parser.add_argument('--label_smoothing', type = float, default = 0.0, help = 'The label smoothing parameter')
    # emb_size of the embedding layer
    parser.add_argument('--emb_size', type = int, default = None, help = 'The embedding size of the model')

    args = parser.parse_args()

    with open('paths.yaml', 'r') as file:
        path_config = yaml.safe_load(file)

    config = Config()

    train_data_path = get_paths(path_config, config.strategy, config.predict_procedure, config.predict_procedure, train = True, processed_data = True)

    _, ids_to_types_map, tokens_to_ids_map, ids_to_tokens_map = load_data(train_data_path['train_data_path'], train = True)

    source_sequences, target_sequences, _ , new_to_old_ids_target = load_data(train_data_path['processed_data_path'], processed_data = True)

    train, test, val = train_test_val_split(source_sequences, target_sequences, test_size = 0.05, valid_size = 0.05, random_state = config.seed)

    train_set  = patientTrajectoryForcastingDataset(**train)
    test_set  = patientTrajectoryForcastingDataset(**test)
    val_set  = patientTrajectoryForcastingDataset(**val)

    old_to_new_ids_target = {v: k for k, v in new_to_old_ids_target.items()}

    target_tokens_to_ids_map = {token: old_to_new_ids_target[idx] for token, idx in tokens_to_ids_map.items() if idx in old_to_new_ids_target.keys()}

    data = get_optimal_embedding_size(source_sequences, target_sequences)

    config.source_vocab_size = data['embedding_size_source']
    config.target_vocab_size = data['embedding_size_target']

    config.seed = enforce_reproducibility(use_seed = None) # input use_seed to have a more deterministic but slower version  

    if args.dry_run:
        os.environ['WANDB_MODE'] = 'dryrun'

    train_dataloader =   DataLoader(train_set, batch_size = config.train_batch_size, num_workers = 2, shuffle = True)
    val_dataloader = DataLoader(test_set, batch_size = config.eval_batch_size, num_workers = 2)

    train_transformer(config, train_dataloader, val_dataloader)
    
    