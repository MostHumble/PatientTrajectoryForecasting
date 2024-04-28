from dataclasses import dataclass, asdict
import yaml
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR, CosineAnnealingWarmRestarts
from utils.train import train_epoch, evaluate, get_data_loaders
import wandb
import argparse
import os
from model import Seq2SeqTransformer


#train_batch_size = 128, eval_batch_size = 128, num_workers = 5,pin_memory = True

@dataclass
class DataConfig:
    strategy : str = 'SDP'
    seed : int = 89957
    test_size : float = 0.05
    valid_size : float = 0.05
    predict_procedure : bool = None
    predict_drugs : bool = None

@dataclass
class Config:
    input_max_length :int = 448
    target_max_length :int = 64
    source_vocab_size : int = None
    target_vocab_size : int = None
    num_encoder_layers: int = 5
    num_decoder_layers: int = 5
    nhead: int = 8
    emb_size: int = 512
    ffn_hid_dim: int = 2048
    train_batch_size: int = 32
    eval_batch_size: int = 256
    learning_rate: float = 3e-4
    warmup_start: float = 5
    num_train_epochs: int = 25
    warmup_epochs: int = None
    label_smoothing : float = 0.0
    target_pad_id : int = 0
    source_pad_id : int = 0

def train_transformer(config, train_dataloader, val_dataloader):

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transformer = Seq2SeqTransformer(config.num_encoder_layers, config.num_decoder_layers, config.emb_size,
                                 config.nhead, config.source_vocab_size,
                                 config.target_vocab_size, config.ffn_hid_dim)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index= config.target_pad_id, label_smoothing = config.label_smoothing)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=config.learning_rate)

    # add wandb loss logging
    for epoch in range(config.num_train_epochs):
        train_loss = train_epoch(transformer,  optimizer, train_dataloader, loss_fn, config.source_pad_id, config.target_pad_id, DEVICE)
        wandb.log({'train.loss': train_loss})
        val_loss =  evaluate(transformer,  optimizer, val_dataloader, loss_fn, config.source_pad_id, config.target_pad_id, DEVICE)
        wandb.log({"Epoch": epoch, "train_loss": train_loss,"val_loss":val_loss, "lr" : optimizer.param_groups[0]['lr']})
    



if __name__ == '__main__':

    config = Config()
    data_config = DataConfig()
    train_dataloader, val_dataloader, test_dataloader, src_tokens_to_ids, tgt_tokens_to_ids, _, data_and_properties  = get_data_loaders(**asdict(data_config))
    
    config.source_vocab_size = data_and_properties['embedding_size_source']
    config.target_vocab_size = data_and_properties['embedding_size_target']
    config.target_pad_id = tgt_tokens_to_ids['PAD']
    config.source_pad_id = src_tokens_to_ids['PAD']

    wandb.init(
      # Set the project where this run will be logged
      project="PTF_SDP_D", 
      # Track hyperparameters and run metadata
      config=asdict(config))
    
    train_transformer(config, train_dataloader, val_dataloader)

    wandb.finish()
    
    