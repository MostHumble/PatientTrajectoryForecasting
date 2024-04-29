from dataclasses import dataclass, asdict
import yaml
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
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
    test_size : float = 0.10
    valid_size : float = 0.05
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
    num_decoder_layers: int = 10
    nhead: int = 8
    emb_size: int = 768
    ffn_hid_dim: int = 1024
    train_batch_size: int = 4
    eval_batch_size: int = 16
    learning_rate: float = 0.0001
    warmup_start: float = 5
    num_train_epochs: int = 45
    warmup_epochs: int = None
    label_smoothing : float = 0.05
    scheduler : str = 'CosineAnnealingWarmRestarts'
    factor : float = 0.1
    patience : int = 5
    T_0 : int = 10
    T_mult : int = 2

    

def train_transformer(config,data_config, train_dataloader, val_dataloader):

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transformer = Seq2SeqTransformer(config.num_encoder_layers, config.num_decoder_layers, config.emb_size,
                                 config.nhead, data_config.source_vocab_size,
                                 data_config.target_vocab_size, config.ffn_hid_dim)
    
    print(f'number of params: {sum(p.numel() for p in transformer.parameters())}')

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        transformer = nn.DataParallel(transformer)

    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index= data_config.target_pad_id, label_smoothing = config.label_smoothing)

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=config.learning_rate)

    # Select the scheduler based on configuration
    if config.scheduler == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    elif config.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=config.factor, patience=config.patience)
    elif config.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.T_0, T_mult=config.T_mult)

    # add wandb loss logging
    for epoch in range(config.num_train_epochs):
        train_loss = train_epoch(transformer,  optimizer, train_dataloader, loss_fn, data_config.source_pad_id, data_config.target_pad_id, DEVICE)
        val_loss =  evaluate(transformer, val_dataloader, loss_fn, data_config.source_pad_id, data_config.target_pad_id, DEVICE)
        wandb.log({"Epoch": epoch, "train_loss": train_loss,"val_loss":val_loss, "lr" : optimizer.param_groups[0]['lr']})
        if config.scheduler :
        # Step the scheduler based on its type
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()



if __name__ == '__main__':

    config = Config()
    data_config = DataConfig()
    train_dataloader, val_dataloader, test_dataloader, src_tokens_to_ids, tgt_tokens_to_ids, _, data_and_properties  = get_data_loaders(**asdict(data_config))
    
    data_config.source_vocab_size = data_and_properties['embedding_size_source']
    data_config.target_vocab_size = data_and_properties['embedding_size_target']
    data_config.target_pad_id = tgt_tokens_to_ids['PAD']
    data_config.source_pad_id = src_tokens_to_ids['PAD']

    wandb.init(
      # Set the project where this run will be logged
      project="PTF_SDP_D", 
      # Track hyperparameters and run metadata
      config=asdict(config))
    try:
        train_transformer(config, data_config, train_dataloader, val_dataloader)
    except Exception as e:
        wandb.log({"error": str(e)})
        raise e
    finally:
        wandb.finish()
    
    