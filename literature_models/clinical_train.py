import itertools
import logging
import os
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from Clinical_GAN.models import Decoder, Discriminator, Encoder, Generator
from datasets import load_from_disk
from PatientTrajectoryForecasting.utils.utils import (
    get_paths,
    load_data,
)
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class FindLR(_LRScheduler):
 
    def __init__(self, optimizer, max_steps, max_lr=10):
        self.max_steps = max_steps
        self.max_lr = max_lr
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * ((self.max_lr / base_lr) ** (self.last_epoch / (self.max_steps - 1)))
                for base_lr in self.base_lrs]


class NoamLR(_LRScheduler):

    def __init__(self, optimizer, warmup_steps,factor =1,model_size=256):
        self.warmup_steps = warmup_steps
        self.model_size = model_size
        self.factor = factor
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        #scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        scale = self.factor * (self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5)))
        #scale = self.factor * (self.model_size ** (-0.5) * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5)))
        return [base_lr * scale for base_lr in self.base_lrs]


def linear_combination(x, y, epsilon): 
    return epsilon*x + (1-epsilon)*y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss

# https://github.com/pytorch/pytorch/issues/7455

#  implementation of Label smoothing with NLLLoss and ignore_index
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon:float=0.1, reduction='mean',ignore_index=-100):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.ignore_index = ignore_index
    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = preds
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction,ignore_index=self.ignore_index)
        return linear_combination(loss/n, nll, self.epsilon)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def padding(pair):
    pair.sort(key=lambda x: len(x[0]), reverse=True)
    inp_batch, output_batch = [],[]
    for pair in pair:
        inp_batch.append(pair[0])
        output_batch.append(pair[1])
    inp= padVar(inp_batch).permute(1,0)
    output = padVar(output_batch).permute(1,0)
    return inp,output

def padVar(inp_batch):
    padList = list(itertools.zip_longest(*inp_batch, fillvalue=0))
    padVar = torch.LongTensor(padList)
    return padVar

def joinrealData(pair):
    data = []
    #print(f"pair : {pair} ")
       
    for pair in pair:
        data.append(pair[0][:-1]+pair[1][1:])
            
    data.sort(key=lambda x: len(x), reverse=True)
    return padVar(data).permute(1,0)

def joinfakeData(pair,output):
    data = []
    #print(f"pair : {pair , len(pair)} output = {output, type(output) , len(output) } ")

    for i in range(len(pair)):
        #print(f"iteration : {i} \n X : {pair[i][0][:-1]} \n Yhat: {output[i]}")
        data.append(pair[i][0][:-1] + output[i])

            
    data.sort(key=lambda x: len(x), reverse=True)
    return padVar(data).permute(1,0)


def convertOutput(pair,reverseOutTypes):
    newPair = []
    for pair in pair:
        newOutput = []
        for code in pair[1]:
            newOutput.append(reverseOutTypes[code])
        newPair.append((pair[0],newOutput))
    return newPair
        
def convertGenOutput(output,reverseOutTypes):
    newOutputs = []
    for codes in output:
        newOutput = []
        for code in codes:
            #print(f" code :{code} output: {output}")
            newOutput.append(reverseOutTypes[code])
        newOutputs.append(newOutput)
        
    return newOutputs   


def unpad(src: torch.Tensor, trg:torch.Tensor) -> Tuple[List,List] :
    sources, targets = [],[]
    for i in range(src.size(0)): # i.e iter through batch size
        sources.append(src[i][src[i]!=0].tolist())
        targets.append(trg[i][trg[i]!=0].tolist())
    return list(zip(sources, targets))


def get_gen_loss(crit_fake_pred):
    #print("crit_fake_pred shape",crit_fake_pred.shape)
    gen_loss = -1. * torch.mean(crit_fake_pred)
    return gen_loss


def get_crit_loss(crit_fake_pred, crit_real_pred):

    crit_loss =  (-1* torch.mean(crit_fake_pred)) - (-1* torch.mean(crit_real_pred))
    return crit_loss

def evaluate(model, Loader, criterion,device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in Loader:
            src,trg = batch['source_sequences'].to(device),batch['target_sequences'].to(device)
            output, _ = model(src, trg[:,:-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(Loader)

def make_src_mask(src_pad_idx, src):

    src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)

    return src_mask


# Configure logging
logging.basicConfig(
    filename='training_clinical_script.log',
    filemode='a',  # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

if __name__ == '__main__':

    path = os.path.join("clinical_script","ClinicalGAN.pth")
    # make the clinical_script dir if does not exist
    if not os.path.exists("clinical_script"):
        os.makedirs("clinical_script")


    logging.info("Starting training")
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
        
        source_sequences = torch.stack(source_sequences, dim=0)
        target_sequences = torch.stack(target_sequences, dim=0)

        return {
            'source_sequences': source_sequences,
            'target_sequences': target_sequences,
        }


    with open('PatientTrajectoryForecasting/paths.yaml', 'r') as file:
            path_config = yaml.safe_load(file)

    train_data_path = get_paths(path_config,
                            'SDP',
                            False,
                            False,
                            train = True,
                            processed_data = True,
                            with_notes = True)

    logging.info("Loading data")
    source_sequences, target_sequences, source_tokens_to_ids, target_tokens_to_ids, _, __, hospital_ids_source = load_data(train_data_path['processed_data_path'],
                                                                                                                    processed_data = True, reindexed = True)
    reverseOutTypes = {v:source_tokens_to_ids[k] for k,v in target_tokens_to_ids.items()}

    train_dataset = torch.load('final_dataset/train_dataset.pth')
    val_dataset = torch.load('final_dataset/val_dataset.pth')
    test_dataset = torch.load('final_dataset/test_dataset.pth')


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    # For the embedding vecs
    SOURCE_VOCAB_SIZE = len(source_tokens_to_ids)
    TARGET_VOCAB_SIZE = len(target_tokens_to_ids)

    MAX_INPUT_LEN = 512
    MAX_OUT_LEN = 96

    SRC_PAD_ID = 0
    TARGET_PAD_ID = 0

    #AGNOSTIC
    DROPOUT = 0.1
    PF_DIM = 512

    # Optimizer, scheduler and loss Params
    LR = 4e-4
    WARMUP_STEPS = 30
    FACTOR = 1 # WTF!
    LABEL_SMOOTHING = 0.0 # WTF!


    # MODEL PARAMS
    # GEN
    N_HEAD_GEN = 8
    N_LAYERS_GEN = 3
    HID_DIM = 256

    # DISC
    N_LAYERS_DISC = 1
    N_HEAD_DISC = 4

    logging.info("Models and optimizers initialization")

    # Models init
    enc = Encoder(input_dim=SOURCE_VOCAB_SIZE, hid_dim=HID_DIM, n_layers=N_LAYERS_GEN, n_heads=N_HEAD_GEN,
                pf_dim=PF_DIM, dropout=DROPOUT, max_length=MAX_INPUT_LEN).to(device)

    dec = Decoder(output_dim=TARGET_VOCAB_SIZE, hid_dim=HID_DIM, n_layers=N_LAYERS_GEN,
                n_heads=N_HEAD_GEN, pf_dim=PF_DIM, dropout=DROPOUT, max_length=MAX_OUT_LEN).to(device)

    gen = Generator(enc, dec, src_pad_idx=SRC_PAD_ID, trg_pad_idx=TARGET_PAD_ID).to(device)

    disc = Discriminator(input_dim=SOURCE_VOCAB_SIZE, hid_dim=HID_DIM, n_layers=N_LAYERS_DISC, n_heads=N_HEAD_DISC,
                        pf_dim=PF_DIM, dropout=DROPOUT, src_pad_idx=SRC_PAD_ID, max_length= MAX_INPUT_LEN+MAX_OUT_LEN).to(device)


    # Optimizers

    gen_opt = torch.optim.Adam(gen.parameters(), lr = LR)
    disc_opt = torch.optim.SGD(disc.parameters(), lr = LR)

    lr_schedulerG = NoamLR(gen_opt, warmup_steps=WARMUP_STEPS, factor=FACTOR, model_size=HID_DIM)
    lr_schedulerD = NoamLR(disc_opt, warmup_steps=WARMUP_STEPS, factor=FACTOR, model_size=HID_DIM)


    gen.apply(initialize_weights)
    disc.apply(initialize_weights)


    criterion = LabelSmoothingCrossEntropy(epsilon=LABEL_SMOOTHING, ignore_index=TARGET_PAD_ID)


    n_epochs = 100
    alpha = 0.3
    clip = 0.1
    gen_clip = 1

    crit_repeats = 5

    train_batch_size = 8
    val_batch_size = 512

    trainLoader = DataLoader(train_dataset,
                                    shuffle = True,
                                    batch_size = train_batch_size,
                                    num_workers = int(os.environ["SLURM_CPUS_PER_TASK"]),
                                    pin_memory = True,
                                    collate_fn = custom_collate_fn)

    valLoader = DataLoader(val_dataset,
                                shuffle = False,
                                batch_size = val_batch_size,
                                num_workers = int(os.environ["SLURM_CPUS_PER_TASK"]),
                                pin_memory = True,
                                collate_fn = custom_collate_fn)


    testLoader = DataLoader(test_dataset,
                                shuffle = False,
                                batch_size = val_batch_size,
                                num_workers = int(os.environ["SLURM_CPUS_PER_TASK"]),
                                pin_memory = True,
                                collate_fn = custom_collate_fn)

    best_valid_loss = float('inf')
    vLoss = []
    tLoss = []
    logging.info("Starting training loop")
    for epoch in range(0, n_epochs):
        totalGen = 0
        totalDis = 0
        epoch_loss = 0
        gen.train()
        disc.train()
        lr_schedulerG.step()
        lr_schedulerD.step()
        for batch in tqdm(trainLoader):
            #innerCount = 0 
            #print(batch_size)
            src, trg = batch['source_sequences'].to(device),batch['target_sequences'].to(device)
            ## Update discriminator ##
            DisLoss =0
            for _ in range(crit_repeats):
                disc_opt.zero_grad()
                output, _ = gen(src, trg[:,:-1]) # encoder-decoder returns output, attention
                _,predValues = torch.max(output,2) 
                # make the input and target sequences have the same codification for the same medical codes
                pair = unpad(src,trg)
                real = joinrealData(convertOutput(pair,reverseOutTypes)) 
                fake = joinfakeData(pair,convertGenOutput(predValues.tolist(),reverseOutTypes))
                #print(f"real : {real.shape} \n fake : {fake.shape}  \n predValues:{predValues}")
                fake_mask =  make_src_mask(0, fake)
                real_mask = make_src_mask(0, real)
                real, fake, fake_mask, real_mask = real.to(device), fake.to(device) , fake_mask.to(device), real_mask.to(device)

                crit_fake_pred = disc(fake,fake_mask)
                crit_real_pred = disc(real, real_mask)
                disc_loss = get_crit_loss(crit_fake_pred, crit_real_pred)
                DisLoss += disc_loss.item()/crit_repeats
                disc_loss.backward(retain_graph=True)
                disc_opt.step()

                for parameters in disc.parameters():
                    parameters.data.clamp_(-clip, clip)
                    
            totalDis += DisLoss
            ## Update generator ##
            gen_opt.zero_grad()
            output, _ = gen(src, trg[:,:-1])
            _,predValues = torch.max(output,2)
            fake = joinfakeData(pair,convertGenOutput(predValues.tolist(),reverseOutTypes))
            fake_mask = make_src_mask(0, fake)
            fake, fake_mask =fake.to(device) , fake_mask.to(device)
            #print(f"gen training fake :{predValues}")
            disc_fake_pred = disc(fake,fake_mask)
            gen_loss1 = get_gen_loss(disc_fake_pred)

            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trgs = trg[:,1:].contiguous().view(-1)

            gen_loss2 = criterion(output,trgs)
            gen_loss = (alpha * gen_loss1)  +  gen_loss2
            totalGen += gen_loss.item()
            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), gen_clip)
            gen_opt.step()
            #epoch_loss = gen_loss.item() + disc_loss.item()
        
        valid_loss = evaluate(gen, valLoader, criterion,device)
        vLoss.append(valid_loss)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # storing the  model which has the least validation loss
            torch.save({'gen_state_dict': gen.state_dict(),
                'disc_state_dict': disc.state_dict(),
                'gen_optimizer_state_dict': gen_opt.state_dict(),
                'disc_optimizer_state_dict': disc_opt.state_dict(),
                'lr':lr_schedulerG.get_last_lr()[0],
                'tLoss':tLoss,
                'vLoss':vLoss}, path)
            print('new best at epoch', epoch)
            logging.info(f'New best at epoch {epoch}')
                    
        tLoss.append(totalGen/len(trainLoader))
        epoch_loss = totalDis + totalGen
        

        print(f'current learning rate : {lr_schedulerG.get_last_lr()}')
        #print(f'current learning rate Discriminator : {lr_schedulerD.get_last_lr()}')
        print(f'Epoch: {epoch+1:02}')
        print(f" Train loss {totalGen/len(trainLoader)} , validation loss :{valid_loss}")
        logging.info(f'Current learning rate: {lr_schedulerG.get_last_lr()}')
        logging.info(f'Epoch: {epoch + 1:02}')
        logging.info(f'Train loss: {totalGen / len(trainLoader)}, Validation loss: {valid_loss}')

    print('Training completed')
    logging.info('Training completed')