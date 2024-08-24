import json
from typing import List

import torch
import torch.utils
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset as tDataset
from tqdm import tqdm


def compute_metrics(preds: List[int], labels: List[int]):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="micro"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def evaluate_model(
    model, dataloader, bin_lens: List[int] = None, tokenizer=None, device="cuda"
):
    """
    Evaluate the model on the given dataloader
    Args:
    - model: The model to evaluate
    - dataloader: The dataloader to evaluate the model on
    - bin_lens: The lengths to split the data into to have a more fine-grained evaluation
    - device: The device to run the model on

    Returns:
    - A dictionary containing the evaluation metrics
    """
    criterion = CrossEntropyLoss()
    sum_loss = 0
    model.eval()
    preds_list = []
    labels_list = []
    loophole = tqdm(dataloader, position=0, leave=True)
    with torch.inference_mode():
        if bin_lens and tokenizer:
            special_token_ids_set = [
                tokenizer.cls_token_id,
                tokenizer.sep_token_id,
                tokenizer.mask_token_id,
                tokenizer.unk_token_id,
            ]
            splits = {
                f"{bin_lens[i]}_{bin_lens[i+1]}_split": {
                    "preds_list": [],
                    "labels_list": [],
                }
                for i in range(len(bin_lens) - 1)
            }
            splits.update(
                {
                    f"sup_{bin_lens[len(bin_lens)-1]}_split": {
                        "preds_list": [],
                        "labels_list": [],
                    }
                }
            )
            metrics_lenghts = {}
            for batch in loophole:
                # Move batch to the device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                labels = batch["labels"].to(device)

                # labels_list.extend(batch['labels'].tolist())
                # Forward pass
                outputs = model.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                )
                pooled_output = outputs[1]
                pooled_output = model.dropout(pooled_output)
                logits = model.classifier(pooled_output)
                preds = logits.view(-1, model.num_labels).argmax(-1)

                lenghts_filter = torch.tensor(
                    [
                        sum(
                            1 if token_id not in special_token_ids_set else 0
                            for token_id in token_ids_seq[attention_mask_seq.bool()]
                        )
                        for token_ids_seq, attention_mask_seq in zip(
                            input_ids, attention_mask
                        )
                    ]
                )
                for bin_, key in zip(bin_lens[1:], splits.keys()):
                    lenghts_filter_ = lenghts_filter < bin_
                    splits[key]["preds_list"].extend(preds[lenghts_filter_].tolist())
                    splits[key]["labels_list"].extend(labels[lenghts_filter_].tolist())
                    lenghts_filter, preds, labels = (
                        lenghts_filter[~lenghts_filter_],
                        preds[~lenghts_filter_],
                        labels[~lenghts_filter_],
                    )

                splits[f"sup_{bin_lens[-1]}_split"]["preds_list"].extend(preds.tolist())
                splits[f"sup_{bin_lens[-1]}_split"]["labels_list"].extend(
                    labels.tolist()
                )

                # preds_list.extend(preds)

                # loss = criterion(logits.view(-1, model.num_labels), labels.view(-1))
                # Update progress bar
                # sum_loss += loss.item()
            for key in splits.keys():
                metrics_lenghts[key] = compute_metrics(
                    splits[key]["preds_list"], splits[key]["labels_list"]
                )

            return metrics_lenghts  # , sum_loss / len(dataloader)

        else:
            for batch in loophole:
                # Move batch to the device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                labels = batch["labels"].to(device)
                labels_list.extend(batch["labels"].tolist())
                # Forward pass
                outputs = model.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                )
                pooled_output = outputs[1]
                pooled_output = model.dropout(pooled_output)
                logits = model.classifier(pooled_output)
                preds_list.extend(logits.view(-1, model.num_labels).argmax(-1).tolist())

                loss = criterion(logits.view(-1, model.num_labels), labels.view(-1))
                # Update progress bar
                sum_loss += loss.item()
    return compute_metrics(preds_list, labels_list), sum_loss / len(dataloader)


# Load the MedNLI dataset
def load_mednli(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    data = [json.loads(line) for line in lines]
    return data


# Convert to Hugging Face datasets
def convert_to_dataset(data):
    return Dataset.from_dict(
        {
            "premise": [item["sentence1"] for item in data],
            "hypothesis": [item["sentence2"] for item in data],
            "label": [
                (
                    0
                    if item["gold_label"] == "entailment"
                    else 1 if item["gold_label"] == "contradiction" else 2
                )
                for item in data
            ],
        }
    )


class NLIDataset(tDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "token_type_ids": torch.tensor(item["token_type_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "labels": torch.tensor(item["label"]),
        }
