import argparse
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import wandb
import yaml
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig
from transformers.models.bert.configuration_bert import BertConfig

from model import Seq2SeqTransformerWithNotes
from utils.bert_embeddings import MosaicBertForEmbeddingGeneration
from utils.eval import mapk, recallTop
from utils.train import (
    WarmupStableDecay,
    create_mask,
    create_source_mask,
    generate_square_subsequent_mask,
)
from utils.utils import (
    get_paths,
    load_data,
)


@dataclass
class DataConfig:
    strategy : str = 'SDP'
    seed : int = 213033
    test_size : float = 0.05
    valid_size : float = 0.10
    predict_procedure : bool = None
    predict_drugs : bool = None
    input_max_length :int = 512
    target_max_length :int = 96
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


class ForcastWithNotes(Dataset):
    def __init__(self, source_sequences, target_sequences, hospital_ids, tokenized_notes):
        self.source_sequences = source_sequences
        self.target_sequences = target_sequences
        self.hospital_ids = hospital_ids
        self.tokenized_notes = load_from_disk(tokenized_notes)
    def __len__(self):
        return len(self.source_sequences)
    def __getitem__(self, idx):
        hospital_ids = self.hospital_ids[idx]
        hospital_ids_lens = len(hospital_ids)

        return  {'source_sequences':torch.tensor(self.source_sequences[idx]),
                 'target_sequences': torch.tensor(self.target_sequences[idx]),
                 'tokenized_notes':self.tokenized_notes[hospital_ids],
                 'hospital_ids_lens': hospital_ids_lens}

def custom_collate_fn(batch):
    source_sequences = [item['source_sequences'] for item in batch]
    target_sequences = [item['target_sequences'] for item in batch]
    tokenized_notes = [item['tokenized_notes'] for item in batch]
    hospital_ids_lens = torch.tensor([item['hospital_ids_lens'] for item in batch], dtype = torch.uint8)
    
    source_sequences = torch.stack(source_sequences, dim=0)
    target_sequences = torch.stack(target_sequences, dim=0)

    # For tokenized_notes, we need to stack each sub-element individually
    tokenized_notes_dict = {key: torch.cat([torch.tensor(tn[key]) for tn in tokenized_notes], dim=0)
                            for key in tokenized_notes[0].keys()}

    return {
        'source_sequences': source_sequences,
        'target_sequences': target_sequences,
        'tokenized_notes': tokenized_notes_dict,
        'hospital_ids_lens': hospital_ids_lens,
    }

def get_sequences_with_notes(model, dataloader : torch.utils.data.dataloader.DataLoader,  source_pad_id : int = 0,
                   tgt_tokens_to_ids : Dict[str, int] =  None, max_len : int = 150,  DEVICE : str ='cuda:0', non_pad_token : int =  42):
    """
    return relevant forcasted and sequences made by the model on the dataset.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        val_dataloader (torch.utils.data.DataLoader): The validation dataloader.
        source_pad_id (int, optional): The padding token ID for the source input. Defaults to 0.
        DEVICE (str, optional): The device to run the evaluation on. Defaults to 'cuda:0'.
        tgt_tokens_to_ids (dict, optional): A dictionary mapping target tokens to their IDs. Defaults to None.
        max_len (int, optional): The maximum length of the generated target sequence. Defaults to 150.
        non_pad_token (int, optional): The non-padding token ID. Defaults to 42.
    Returns:
        List[List[int]], List[List[int]]: The list of relevant and forecasted sequences.
    """

    model.eval()
    pred_trgs = []
    targets = []
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc='scoring'):
            
            batch_pred_trgs = []
            batch_targets = []
            
            tokenized_notes = batch['tokenized_notes']
            hospital_ids_lens = batch['hospital_ids_lens'].to(DEVICE)
            
            notes_input_ids = tokenized_notes['input_ids'].to(DEVICE)
            notes_token_type_ids = tokenized_notes['token_type_ids'].to(DEVICE)
            notes_attention_mask = tokenized_notes['attention_mask'].to(DEVICE)
            
            # CCS/ICD data
            source_input_ids = batch['source_sequences'].to(DEVICE)
            target_input_ids = batch['target_sequences'].to(DEVICE)
                        
            # just to have the correct mask, don't have time to modify
            if model.bert.config.strategy == 'concat':
                temp_enc = torch.cat([torch.full((source_input_ids.size(0), model.bert.config.num_embedding_layers), non_pad_token, device = DEVICE), source_input_ids], dim = 1)
            elif model.bert.config.strategy == 'mean':
                temp_enc = torch.cat([torch.full((source_input_ids.size(0), 1), non_pad_token, device = DEVICE), source_input_ids], dim = 1)
            elif model.bert.config.strategy == 'all':
                temp_enc = []
                for i,length in enumerate(hospital_ids_lens):
                    # we cat across seq_len, and truncate to 512 (max_seq_len_input)
                    temp_enc.append(torch.cat([torch.full((length.item() * model.bert.config.num_embedding_layers,), non_pad_token, device = DEVICE), source_input_ids[i]], dim = 0)[:512])
                # batch, seq_len  
                temp_enc = torch.stack(temp_enc)
                
            # concat across seq_len
            source_mask, source_padding_mask = create_source_mask(temp_enc, source_pad_id, DEVICE)
                                                                                                       
            del temp_enc
            
            memory = model.batch_encode(src = source_input_ids,
                                        src_mask = source_mask,
                                        src_key_padding_mask = source_padding_mask,
                                        notes_input_ids = notes_input_ids,
                                        notes_attention_mask = notes_attention_mask,
                                        notes_token_type_ids = notes_token_type_ids,
                                        hospital_ids_lens = hospital_ids_lens
                                       )
            
            pred_trg = torch.tensor(tgt_tokens_to_ids['BOS'], device= DEVICE).repeat(source_input_ids.size(0)).unsqueeze(1)
            # generate target sequence one token at a time at batch level
            for i in range(max_len):
                trg_mask = generate_square_subsequent_mask(i+1, DEVICE)
                output = model.decode(pred_trg, memory, trg_mask)
                probs = model.generator(output[:, -1])
                pred_tokens = torch.argmax(probs, dim=1)
                pred_trg = torch.cat((pred_trg, pred_tokens.unsqueeze(1)), dim=1)
                eov_mask = pred_tokens == tgt_tokens_to_ids['EOV']

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


def evaluate_with_notes(model, val_dataloader, loss_fn,  source_pad_id = 0, target_pad_id = 0, DEVICE='cuda:0', non_pad_token = 42):
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
        
        for batch in tqdm(val_dataloader, desc='evaluation'):
            
                # notes data
            tokenized_notes = batch['tokenized_notes']
            hospital_ids_lens = batch['hospital_ids_lens'].to(DEVICE)
            
            notes_input_ids = tokenized_notes['input_ids'].to(DEVICE)
            notes_token_type_ids = tokenized_notes['token_type_ids'].to(DEVICE)
            notes_attention_mask = tokenized_notes['attention_mask'].to(DEVICE)
            
            # CCS/ICD data
            source_input_ids = batch['source_sequences'].to(DEVICE)
            target_input_ids = batch['target_sequences'].to(DEVICE)
            
            target_input_ids_ = target_input_ids[:, :-1]
            
            # just to have the correct mask, don't have time to modify
            # source_input_ids : batch , seq_len
            if model.bert.config.strategy == 'concat':
                temp_enc = torch.cat([torch.full((source_input_ids.size(0), model.bert.config.num_embedding_layers), non_pad_token, device = DEVICE), source_input_ids], dim = 1)
            elif model.bert.config.strategy == 'mean':
                temp_enc = torch.cat([torch.full((source_input_ids.size(0), 1), non_pad_token, device = DEVICE), source_input_ids], dim = 1)
            elif model.bert.config.strategy == 'all':
                temp_enc = []
                for i,length in enumerate(hospital_ids_lens):
                    # we cat across seq_len, and truncate to 512 (max_seq_len_input)
                    temp_enc.append(torch.cat([torch.full((length.item() * model.bert.config.num_embedding_layers,), non_pad_token, device = DEVICE), source_input_ids[i]], dim = 0)[:512])
                # batch, seq_len  
                temp_enc = torch.stack(temp_enc)
                
            # concat across seq_len
            source_mask, target_mask, source_padding_mask, target_padding_mask = create_mask(temp_enc,
                                                                                             target_input_ids_,
                                                                                             source_pad_id,
                                                                                             target_pad_id,
                                                                                             DEVICE)
            del temp_enc
            
    
            logits = model(src = source_input_ids,
                             trg = target_input_ids_,
                             src_mask = source_mask,
                             tgt_mask = target_mask,
                             src_padding_mask = source_padding_mask,
                             tgt_padding_mask = target_padding_mask,
                             memory_key_padding_mask = source_padding_mask,
                             notes_input_ids = notes_input_ids,
                             notes_attention_mask = notes_attention_mask,
                             notes_token_type_ids = notes_token_type_ids,
                             hospital_ids_lens = hospital_ids_lens
                            )
    
            _target_input_ids = target_input_ids[:, 1:]
            
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), _target_input_ids.reshape(-1))
            losses += loss.item()

    return losses / len(val_dataloader)

def xavier_init(transformer):

    for p in transformer.transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    for p in transformer.generator.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    for p in transformer.src_tok_emb.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    for p in transformer.tgt_tok_emb.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    if transformer.bert.config.strategy == 'all':
        for p in transformer.projection.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    return transformer
def train_epoch_with_notes(model, scheduler , optimizer, train_dataloader, loss_fn, source_pad_id = 0, target_pad_id = 0, DEVICE='cuda:0', non_pad_token = 42):
    
    losses = 0
    
    for batch in tqdm(train_dataloader, desc='train'):
        
        model.train()
        optimizer.zero_grad()

    
        # notes data
        tokenized_notes = batch['tokenized_notes']
        hospital_ids_lens = batch['hospital_ids_lens'].to(DEVICE)
        
        notes_input_ids = tokenized_notes['input_ids'].to(DEVICE)
        notes_token_type_ids = tokenized_notes['token_type_ids'].to(DEVICE)
        notes_attention_mask = tokenized_notes['attention_mask'].to(DEVICE)
        
        # CCS/ICD data
        source_input_ids = batch['source_sequences'].to(DEVICE)
        target_input_ids = batch['target_sequences'].to(DEVICE)
        
        target_input_ids_ = target_input_ids[:, :-1]
        
        # just to have the correct mask, don't have time to modify
        if model.bert.config.strategy == 'concat':
            temp_enc = torch.cat([torch.full((source_input_ids.size(0), model.bert.config.num_embedding_layers), non_pad_token, device = DEVICE), source_input_ids], dim = 1)
        elif model.bert.config.strategy == 'mean':
            temp_enc = torch.cat([torch.full((source_input_ids.size(0), 1), non_pad_token, device = DEVICE), source_input_ids], dim = 1)
        elif model.bert.config.strategy == 'all':
            temp_enc = []
            for i,length in enumerate(hospital_ids_lens):
                # we cat across seq_len, and truncate to 512 (max_seq_len_input)
                # we do this because we cat from mulitple layers of the bert encoder * num_visits
                temp_enc.append(torch.cat([torch.full((length.item() * model.bert.config.num_embedding_layers,), non_pad_token, device = DEVICE), source_input_ids[i]], dim = 0)[:512])
            # batch, seq_len  
            temp_enc = torch.stack(temp_enc)
            
        # concat across seq_len
        source_mask, target_mask, source_padding_mask, target_padding_mask = create_mask(temp_enc,
                                                                                         target_input_ids_,
                                                                                         source_pad_id,
                                                                                         target_pad_id,
                                                                                         DEVICE)
        del temp_enc
        
        logits = model(src = source_input_ids,
                             trg = target_input_ids_,
                             src_mask = source_mask,
                             tgt_mask = target_mask,
                             src_padding_mask = source_padding_mask,
                             tgt_padding_mask = target_padding_mask,
                             memory_key_padding_mask = source_padding_mask,
                             notes_input_ids = notes_input_ids,
                             notes_attention_mask = notes_attention_mask,
                             notes_token_type_ids = notes_token_type_ids,
                             hospital_ids_lens = hospital_ids_lens
                            )
        
        _target_input_ids = target_input_ids[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), _target_input_ids.reshape(-1))
        loss.backward()
        optimizer.step()
        losses += loss.item()

        if scheduler[1] == 'WarmupStableDecay':
                scheduler[0].step()
            
    return losses / len(train_dataloader)




def get_model (
    pretrained_model_name: str = 'bert-base-uncased',
    model_config: Optional[dict] = None,
    pretrained_checkpoint: Optional[str] = None,
    num_embedding_layers : int = 4,
    strategy = 'concat',
    seq_len = 512
    ):
    
    model_config, unused_kwargs = BertConfig.get_config_dict(model_config)
    model_config.update(unused_kwargs)
    
    config, unused_kwargs = AutoConfig.from_pretrained(
        pretrained_model_name, return_unused_kwargs=True, **model_config)
    # This lets us use non-standard config fields (e.g. `starting_alibi_size`)
    config.update(unused_kwargs)
    config.num_embedding_layers = num_embedding_layers
    config.strategy = strategy
    config.seq_len = seq_len
    model = MosaicBertForEmbeddingGeneration.from_pretrained(
            pretrained_checkpoint=pretrained_checkpoint, config=config)
    
    return model




def train_transformer(args, data_config, train_dataloader, val_dataloader, test_dataloader, ks = [20,40,60], positional_encoding = True, dropout = 0.1, DEVICE='cuda:0'):

    bert_model = get_model(pretrained_model_name=PRETRAINED_MODEL_NAME,
                           model_config=MODEL_CONFIG,
                           pretrained_checkpoint=PRETRAINED_MODEL_CHECKPOINT,
                           num_embedding_layers=args.num_embedding_layers,
                           strategy=args.strategy)


    transformer = Seq2SeqTransformerWithNotes(num_encoder_layers = args.num_encoder_layers,
                                              num_decoder_layers = args.num_decoder_layers,
                                              emb_size = args.emb_size,
                                              nhead = args.nhead,
                                              src_vocab_size = data_config.source_vocab_size,
                                              tgt_vocab_size = data_config.target_vocab_size,
                                              use_positional_encoding_notes = args.use_positional_encoding_notes,
                                              dim_feedforward = args.ffn_hid_dim,
                                              dropout = dropout,
                                              positional_encoding = positional_encoding,
                                              bert = bert_model)

    for param in transformer.bert.parameters():
        param.requires_grad = False

    print(f'number of params: {sum(p.numel() for p in transformer.parameters())/1e6 :.2f}M')

    transformer = xavier_init(transformer)

    transformer = transformer.to(DEVICE)
   
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index= data_config.target_pad_id, label_smoothing = args.label_smoothing)

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=args.learning_rate)

    # Select the scheduler based on configuration
    if args.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    elif args.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 4, eta_min = 1e-6)
    elif args.scheduler == 'OneCycleLR':
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, total_steps=args.num_train_epochs)
    elif args.scheduler == 'WarmupStableDecay':
        total_steps = len(train_dataloader) * args.num_train_epochs
        num_warmup_steps = int(total_steps * args.warmup_ratio)
        num_stable_steps = int(total_steps * args.stable_ratio)
        num_decay_steps = total_steps - num_warmup_steps - num_stable_steps
        scheduler = WarmupStableDecay(optimizer, num_warmup_steps = num_warmup_steps, num_stable_steps = num_stable_steps, num_decay_steps = num_decay_steps, min_lr_ratio = args.min_lr_ratio)
    
    # add wandb loss logging
    for epoch in range(args.num_train_epochs):
        test_mapk = {}
        # model, optimizer, train_dataloader, loss_fn, source_pad_id = 0, target_pad_id = 0, DEVICE='cuda:0', non_pad_token = 42
        train_loss = train_epoch_with_notes(model = transformer,
                                            optimizer = optimizer,
                                            scheduler = [scheduler, args.scheduler],
                                            train_dataloader = train_dataloader,
                                            loss_fn = loss_fn,
                                            source_pad_id = data_config.source_pad_id,
                                            target_pad_id = data_config.target_pad_id,
                                            DEVICE = DEVICE,
                                            non_pad_token = 42
                                            )
        print(f'Train loss: {train_loss}', flush=True)
        if epoch  % args.eval_every == 0 and epoch != 0:
            val_loss =  evaluate_with_notes(transformer, val_dataloader, loss_fn, data_config.source_pad_id, data_config.target_pad_id, DEVICE)
            pred_trgs, targets =  get_sequences_with_notes(transformer, test_dataloader, data_config.source_pad_id, target_tokens_to_ids, max_len = 96, DEVICE = DEVICE)
            if pred_trgs:
                test_mapk = {f"test_map@{k}": mapk(targets, pred_trgs, k) for k in ks}
                test_recallk = {f"test_recall@{k}": recallTop(targets, pred_trgs, rank = [k])[0] for k in ks}
                wandb.log({"Epoch": epoch, "train_loss": train_loss,"val_loss":val_loss, "lr" : optimizer.param_groups[0]['lr'], **test_mapk, **test_recallk}, step = epoch)
        else:
            wandb.log({"Epoch": epoch, "train_loss": train_loss, "lr" : optimizer.param_groups[0]['lr']})
        if args.scheduler == 'ReduceLROnPlateau':
            scheduler.step(train_loss)
        elif args.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler.step()

if __name__ == '__main__':

    PRETRAINED_MODEL_CHECKPOINT = os.path.join('bert_mimic_model_512/step_80000', 'pytorch_model.bin')
    PRETRAINED_MODEL_NAME = 'mosaicml/mosaic-bert-base-seqlen-512'
    MODEL_CONFIG = 'mosaicml/mosaic-bert-base-seqlen-512'


    parser = argparse.ArgumentParser(description='CLI for wandb sweep parameters')


    # Fixed value parameters

    # Integer uniform distribution parameters
    
    parser.add_argument('--ffn_hid_dim', type=int, default = 3072, help='Hidden dimension size of feed forward network (min 512, max 4096)')
    parser.add_argument('--emb_size', type=int, default = 768, help='Embedding size (min 128, max 1024)')
    parser.add_argument('--num_decoder_layers', default= 12,type=int, help='Number of decoder layers (min 6, max 16)')
    parser.add_argument('--num_encoder_layers', type=int, default= 8 ,help='Number of encoder layers (min 6, max 16)')
    parser.add_argument('--num_train_epochs', type=int, default=30, help='Number of training epochs (min 13, max 100)')
    parser.add_argument('--nhead', type=int, default=8, help='Number of heads (min 4, max 16)')
    # Uniform distribution parameters
    parser.add_argument('--label_smoothing', type=float, default = 0.09, help='Label smoothing (min 0, max 0.2)')
    parser.add_argument('--learning_rate', type=float, default = 3e-5, help='Learning rate (min 5e-05, max 0.008)')

    # Categorical distribution parameters
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingWarmRestarts', choices=['ReduceLROnPlateau', 'CosineAnnealingWarmRestarts', 'WarmupStableDecay'], help='Type of scheduler')

    # Quantized log uniform distribution parameters (handled as int for simplicity)
    parser.add_argument('--train_batch_size', type=int, default = 16, help='Training batch size (min 32, max 48)')

    parser.add_argument('--strategy', type=str, default ='concat',help='Strategy for embedding generation')
    parser.add_argument('--num_embedding_layers', type=int, default = 6, help='Number of layers to use for embedding generation')
    parser.add_argument('--predict_procedure', type=bool, default = False, help='Predict procedure codes')
    parser.add_argument('--predict_drugs', type=bool, default = False, help='Predict drug codes')
    parser.add_argument('--ptf_strategy', type=str, default = 'SDP', help='Strategy for patient trajectory forecasting')
    parser.add_argument('--use_positional_encoding_notes', type=str, default = 'False', help='Use positional encoding for notes')
    parser.add_argument('--eval_every', type=int, default = 1, help='Evaluate every n epochs')
    parser.add_argument('--warmup_ratio', type=float, default = 0.6, help='Warmup ratio')
    parser.add_argument('--stable_ratio', type=float, default = 0.1, help='Stable ratio')
    parser.add_argument('--min_lr_ratio', type=float, default = 0.2, help='Minimum learning rate ratio')

    args = parser.parse_args()
    if args.use_positional_encoding_notes.lower() == 'true':
        args.use_positional_encoding_notes = True
        print('Using positional encoding for notes')
    else:
        args.use_positional_encoding_notes = False
        print('Not using positional encoding for notes')
        
    config = Config()

    data_config = DataConfig()

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    with open('PatientTrajectoryForecasting/paths.yaml', 'r') as file:
        path_config = yaml.safe_load(file)

    train_data_path = get_paths(path_config,
                            args.ptf_strategy,
                            args.predict_procedure,
                            args.predict_drugs,
                            train = True,
                            processed_data = True,
                            with_notes = True)
    
    source_sequences, target_sequences, source_tokens_to_ids, target_tokens_to_ids, _, __, hospital_ids_source = load_data(train_data_path['processed_data_path'],
                                                                                                                           processed_data = True, reindexed = True)
    # Load the datasets
    train_dataset = torch.load('final_dataset/train_dataset.pth')
    val_dataset = torch.load('final_dataset/val_dataset.pth')
    test_dataset = torch.load('final_dataset/test_dataset.pth')

    
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  batch_size=args.train_batch_size,
                                  num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                                  pin_memory=True,
                                  collate_fn=custom_collate_fn)

    val_dataloader = DataLoader(val_dataset,
                                shuffle = False,
                                batch_size=args.train_batch_size * 2,
                                num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                                pin_memory=True,
                                collate_fn=custom_collate_fn)


    test_dataloader = DataLoader(test_dataset,
                                 shuffle = False,
                                 batch_size=args.train_batch_size * 2,
                                 num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                                 pin_memory=True,
                                 collate_fn=custom_collate_fn)
    
    
    data_config.source_vocab_size = ((len(source_tokens_to_ids) + 63) // 64) *64
    data_config.target_vocab_size = ((len(target_tokens_to_ids) + 63) // 64) *64

    data_config.target_pad_id = target_tokens_to_ids['PAD']
    data_config.source_pad_id = source_tokens_to_ids['PAD']

    wandb.init(
    # Set the project where this run will be logged
    project="PTF_SDP_D_NOTES_LONGER", config=args)
    
    try:
        train_transformer(args, data_config=data_config,
                          train_dataloader=train_dataloader,
                          val_dataloader=val_dataloader,
                          test_dataloader=test_dataloader,
                          DEVICE=DEVICE)
    except Exception as e:
        wandb.log({"error": str(e)})
    finally:
        wandb.finish()
    
    