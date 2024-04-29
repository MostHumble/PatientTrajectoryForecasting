from dataclasses import dataclass, asdict
import yaml
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from utils.train import train_epoch, evaluate, get_data_loaders
import wandb
import argparse
import os
from tqdm import tqdm
from utils.train import create_mask
# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
import math
from torch import Tensor
import torch.nn as nn
from torch.nn import Transformer



class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
    
# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int,
                 dropout: float = 0.1, device_count = 1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model = emb_size,
                                       nhead = nhead,
                                       num_encoder_layers = num_encoder_layers,
                                       num_decoder_layers = num_decoder_layers,
                                       dim_feedforward = dim_feedforward,
                                       dropout = dropout,
                                       batch_first = True, norm_first = True)
            
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.src_tok_emb(src)
        tgt_emb = self.tgt_tok_emb(trg)
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(
                            self.src_tok_emb(src), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(
                          self.tgt_tok_emb(tgt), memory,
                          tgt_mask)

#train_batch_size = 128, eval_batch_size = 128, num_workers = 5,pin_memory = True
def train_epoch(model, optimizer, train_dataloader, loss_fn, source_pad_id = 0, target_pad_id = 0, DEVICES=['cuda:0', 'cuda:1']):
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
        
        source_input_ids = source_input_ids.to(DEVICES[0])
        target_input_ids = target_input_ids.to(DEVICES[1])
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

def evaluate(model, val_dataloader, loss_fn,  source_pad_id = 0, target_pad_id = 0, DEVICES=['cuda:0', 'cuda:1']):
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

    with torch.no_grad():
        
        for source_input_ids, target_input_ids in tqdm(val_dataloader, desc='evaluation'):
            
            source_input_ids = source_input_ids.to(DEVICES[0])
            target_input_ids = target_input_ids.to(DEVICES[1])
            
            target_input_ids_ = target_input_ids[:, :-1]        

            source_mask, target_mask, source_padding_mask, target_padding_mask = create_mask(source_input_ids, target_input_ids_, source_pad_id, target_pad_id, DEVICE)

            logits = model(source_input_ids, target_input_ids_, source_mask, target_mask, source_padding_mask, target_padding_mask, source_padding_mask)

            _target_input_ids = target_input_ids[:, 1:]
            
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), _target_input_ids.reshape(-1))
            losses += loss.item()

    return losses / len(val_dataloader)

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
    num_encoder_layers: int = 8
    num_decoder_layers: int = 6
    nhead: int = 8
    emb_size: int = 1024
    ffn_hid_dim: int = 1024
    train_batch_size: int = 1
    eval_batch_size: int = 16
    learning_rate: float = 0.0001
    warmup_start: float = 5
    num_train_epochs: int = 45
    warmup_epochs: int = None
    label_smoothing : float = 0.05
    scheduler : str = 'ReduceLROnPlateau'
    factor : float = 0.1
    patience : int = 5

    

def train_transformer(config,data_config, train_dataloader, val_dataloader):

    if torch.cuda.is_available():
        DEVICES = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    else:
        DEVICES = [torch.device("cpu")]

    transformer = Seq2SeqTransformer(config.num_encoder_layers, config.num_decoder_layers, config.emb_size,
                                 config.nhead, data_config.source_vocab_size,
                                 data_config.target_vocab_size, config.ffn_hid_dim)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


    if torch.cuda.device_count() > 1:
        transformer.src_tok_emb.to('cuda:0')
        transformer.transformer.encoder.to('cuda:0')
        transformer.tgt_tok_emb.to('cuda:1')
        transformer.transformer.decoder.to('cuda:1')
        transformer.generator.to('cuda:1')

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
        train_loss = train_epoch(transformer,  optimizer, train_dataloader, loss_fn, data_config.source_pad_id, data_config.target_pad_id, DEVICES)
        val_loss =  evaluate(transformer, val_dataloader, loss_fn, data_config.source_pad_id, data_config.target_pad_id, DEVICES)
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
    
    