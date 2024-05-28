import os
import torch
import triton
from datasets import load_dataset
import evaluate 
from functools import partial
from accelerate import Accelerator
from transformers.models.bert.configuration_bert import BertConfig

from transformers import (
    AutoModelForMaskedLM,
    BertTokenizer,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainingArguments
    )
from utils.notes import (
    _tokenize_function,
    _compute_metrics,
    preprocess_logits_for_metrics
)
# import partial functools
# set up the tokenize_function, to take a the default tokenizer param with partial


if __name__ == "__main__":

    DEBUG = False
    NUM_EPOCHS = 50
    TRAIN_BATCH_SIZE = 176


    accelerator = Accelerator()
        
    cache_dir = "/scratch/sifal.klioui/cache"
    os.makedirs(cache_dir, exist_ok = True)

    train_path = "/scratch/sifal.klioui/notes/train/*.txt"
    val_path = "/scratch/sifal.klioui/notes/validation/*.txt"

    raw_datasets = load_dataset(path ='/scratch/sifal.klioui/notes/', data_files={"train": train_path, "validation":val_path}, cache_dir = cache_dir)


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', use_fast_tokenizer = True, cache_dir = cache_dir)

    tokenize_function = partial(_tokenize_function, tokenizer = tokenizer)

    # for mulitprocessing, we need to use the accelerator context manager
    with accelerator.main_process_first():

        tokenized_datasets = raw_datasets.map(
                            tokenize_function,
                            batched = True,
                            num_proc = 9,
                            remove_columns=['text'],
                            load_from_cache_file= True,
                            desc="Running tokenizer on dataset line_by_line",
                        )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    config = BertConfig.from_pretrained('mosaicml/mosaic-bert-base')

    model = AutoModelForMaskedLM.from_pretrained('mosaicml/mosaic-bert-base', trust_remote_code=True, config = config, cache_dir = cache_dir)

    compute_metrics = partial(_compute_metrics, evaluate.load("accuracy", cache_dir=cache_dir))

    # select a subset of the data for debugging
    if DEBUG:
        train_dataset = train_dataset.select(range(len(train_dataset)//100))
        eval_dataset = eval_dataset.select(range(len(eval_dataset)//100))


    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability= 0.30,
        pad_to_multiple_of= 8 
    )

    os.environ["WANDB_PROJECT"] = "bert_test"  # name your W&B project

        # Training Arguments
    learning_rate =  5.0e-4

    train_steps = len(train_dataset) // TRAIN_BATCH_SIZE * NUM_EPOCHS
    warmup_steps = int(train_steps * 0.05)

    training_args = TrainingArguments(
        save_safetensors = False,
        output_dir="./train_mosa",
        eval_strategy = "steps",
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        learning_rate =  5.0e-4,
        adam_beta1 = 0.9,
        adam_beta2 = 0.98,
        adam_epsilon = 1.0e-06,
        weight_decay = 1.0e-5,
        logging_steps=1000,
        save_steps=1000,
        save_total_limit=2,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        report_to="wandb"
    )


        # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )


    trainer.train()

        