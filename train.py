from dataclasses import dataclass, asdict
import yaml
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from utils.train import train_epoch, evaluate, get_data_loaders
import wandb
import argparse
import os
from model import Seq2SeqTransformer, Seq2SeqTransformerWithNotes
from utils.eval import mapk, get_sequences
import warnings
# currently getting warnings because of mask datatypes, you might wanna change this not installing from environment.yml
warnings.filterwarnings("ignore") 

#train_batch_size = 128, eval_batch_size = 128, num_workers = 5,pin_memory = True

@dataclass
class DataConfig:
    strategy : str = 'SDP'
    seed : int = 213033
    test_size : float = 0.05
    valid_size : float = 0.10
    predict_procedure : bool = None
    predict_drugs : bool = None
    input_max_length :int = 448
    target_max_length :int = 64
    source_vocab_size : int = None
    target_vocab_size : int = None
    target_pad_id : int = 0
    source_pad_id : int = 0

@dataclass
class Config:
    num_encoder_layers: int = 12
    num_decoder_layers: int = 8
    nhead: int = 8
    emb_size: int = 768
    positional_encoding : bool = True
    ffn_hid_dim: int = 2048
    dropout: float = 0.1
    train_batch_size: int = 128
    eval_batch_size: int = 256
    learning_rate: float = 0.0001
    num_train_epochs: int = 45
    warmup_epochs: int = None
    label_smoothing : float = 0.05
    scheduler : str = 'CosineAnnealingWarmRestarts'
    factor : float = 0.1
    patience : int = 5
    T_0 : int = 3
    T_mult : int = 2
    step_size : int = 10
    gamma : float = 0.1

    
from torch.nn.parallel import DistributedDataParallel as DDP

def train_transformer(config, data_config, train_dataloader, val_dataloader, ks = [20,40,72], positional_encoding = True, dropout = 0.1, with_notes = False):

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if with_notes:
        transformer = Seq2SeqTransformerWithNotes(config.num_encoder_layers, config.num_decoder_layers,
                                      config.emb_size, config.nhead, 
                                      data_config.source_vocab_size, data_config.target_vocab_size,
                                      config.ffn_hid_dim,
                                      dropout,
                                      positional_encoding)
    else:
        transformer = Seq2SeqTransformer(config.num_encoder_layers, config.num_decoder_layers,
                                        config.dim_per_head * config.nhead, config.nhead, 
                                        data_config.source_vocab_size, data_config.target_vocab_size,
                                        config.ffn_hid_dim,
                                        dropout,
                                        positional_encoding)
    
    print(f'number of params: {sum(p.numel() for p in transformer.parameters())/1e6 :.2f}M')

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index= data_config.target_pad_id, label_smoothing = config.label_smoothing)

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=config.learning_rate)

    # Select the scheduler based on configuration
    if config.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    elif config.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 4, eta_min=1e-6)
    
    # add wandb loss logging
    for epoch in range(config.num_train_epochs):
        val_mapk = {}
        train_loss = train_epoch(transformer,  optimizer, train_dataloader, loss_fn, data_config.source_pad_id, data_config.target_pad_id, DEVICE)
        val_loss =  evaluate(transformer, val_dataloader, loss_fn, data_config.source_pad_id, data_config.target_pad_id, DEVICE)
        pred_trgs, targets =  get_sequences(transformer, val_dataloader, data_config.source_pad_id, tgt_tokens_to_ids, max_len = 96, DEVICE = DEVICE)
        if pred_trgs:
            val_mapk = {f"val_map@{k}": mapk(targets, pred_trgs, k) for k in ks}
            wandb.log({"Epoch": epoch, "train_loss": train_loss,"val_loss":val_loss, "lr" : optimizer.param_groups[0]['lr'], **val_mapk})
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif isinstance(scheduler, lr_scheduler.CosineAnnealingWarmRestarts):
            scheduler.step()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CLI for wandb sweep parameters')


    # Fixed value parameters
    parser.add_argument('--dim_per_head', type=int, default=64, help='Dimension per head')

    # Integer uniform distribution parameters
    parser.add_argument('--ffn_hid_dim', type=int, help='Hidden dimension size of feed forward network (min 512, max 4096)')
    parser.add_argument('--nhead', type=int, help='Number of heads (min 4, max 16)')
    parser.add_argument('--num_decoder_layers', type=int, help='Number of decoder layers (min 6, max 16)')
    parser.add_argument('--num_encoder_layers', type=int, help='Number of encoder layers (min 6, max 16)')
    parser.add_argument('--num_train_epochs', type=int, help='Number of training epochs (min 13, max 100)')

    # Uniform distribution parameters
    parser.add_argument('--label_smoothing', type=float, help='Label smoothing (min 0, max 0.2)')
    parser.add_argument('--learning_rate', type=float, help='Learning rate (min 5e-05, max 0.008)')

    # Categorical distribution parameters
    parser.add_argument('--scheduler', type=str, choices=['ReduceLROnPlateau', 'CosineAnnealingWarmRestarts'], help='Type of scheduler')

    # Quantized log uniform distribution parameters (handled as int for simplicity)
    parser.add_argument('--train_batch_size', type=int, help='Training batch size (min 32, max 64)')

    args = parser.parse_args()

    config = Config()
    data_config = DataConfig()

    train_dataloader, val_dataloader, test_dataloader, src_tokens_to_ids, tgt_tokens_to_ids, _, embedding_sizes = get_data_loaders(train_batch_size=args.train_batch_size, eval_batch_size=args.train_batch_size*2, pin_memory=True, **asdict(data_config))
    
    data_config.source_vocab_size = embedding_sizes['embedding_size_source']
    data_config.target_vocab_size = embedding_sizes['embedding_size_target']
    data_config.target_pad_id = tgt_tokens_to_ids['PAD']
    data_config.source_pad_id = src_tokens_to_ids['PAD']

    wandb.init(
      # Set the project where this run will be logged
      project="PTF_SDP_D", config=asdict(config), reinit=True
      )
    try:
        train_transformer(args, data_config, train_dataloader, val_dataloader)
    except Exception as e:
        wandb.log({"error": str(e)})
        raise e
    finally:
        wandb.finish()
    
    