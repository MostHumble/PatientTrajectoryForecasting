from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch 
import json
from datasets import Dataset
from torch.utils.data import Dataset as tDataset

def compute_metrics(preds, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, 
                                                               preds, 
                                                               average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def evaluate_model(model, dataloader, device, criterion):
    sum_loss = 0 
    model.eval()
    preds_list = []
    labels_list = []
    loophole = tqdm(dataloader, position=0, leave=True)
    with torch.inference_mode():
        for batch in loophole:
            # Move batch to the device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            labels_list.extend(batch['labels'].tolist())
            # Forward pass
            outputs = model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            pooled_output = outputs[1]
            pooled_output = model.dropout(pooled_output)
            logits = model.classifier(pooled_output)
            preds_list.extend(logits.view(-1, model.num_labels).argmax(-1).tolist())
            
            loss = criterion(logits.view(-1, model.num_labels), labels.view(-1))
            #r_scheduler.step()
            # Update progress bar
            sum_loss += loss.item()
    return compute_metrics(preds_list, labels_list), sum_loss / len(dataloader)


# Load the MedNLI dataset
def load_mednli(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = [json.loads(line) for line in lines]
    return data


# Convert to Hugging Face datasets
def convert_to_dataset(data):
    return Dataset.from_dict({
        'premise': [item['sentence1'] for item in data],
        'hypothesis': [item['sentence2'] for item in data],
        'label': [0 if item['gold_label'] == 'entailment' else 1 if item['gold_label'] == 'contradiction' else 2 for item in data]
    })


class NLIDataset(tDataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            'input_ids': torch.tensor(item['input_ids']),
            'token_type_ids': torch.tensor(item['token_type_ids']),
            'attention_mask': torch.tensor(item['attention_mask']),
            'labels': torch.tensor(item['label'])
        }
