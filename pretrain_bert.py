import argparse
import json
import logging
import math
import random
from itertools import chain
from pathlib import Path
from functools import partial
from tqdm import tqdm
import os
import torch
import triton
import transformers
import evaluate

import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers.models.bert.configuration_bert import BertConfig
from transformers.optimization import get_wsd_schedule
from huggingface_hub import HfApi
from torch.optim import AdamW



from transformers import (
    AutoModelForMaskedLM,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType
    )


from utils.notes import (
    _tokenize_function,
    prepare_sequences,
    preprocess_logits_for_metrics
)


logger = get_logger(__name__)


def parse_args():
    
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")

    parser.add_argument("--debug", action='store_true', help="Whether to run the model in debug mode.")
    parser.add_argument("--output_dir", type=str, default='./bert_mimic_model', help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=314159, help="A seed for reproducible training.")

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument("--num_train_epochs", type=int, default=5, help="Total number of training epochs to perform.")

    
    parser.add_argument(
            "--max_seq_length",
            type=int,
            default=128,
            help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."

        )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.30, help="Ratio of tokens to mask for masked language modeling loss"
    )

    parser.add_argument(
        "--trust_remote_code",
        action='store_false',
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    
    parser.add_argument(
        "--with_tracking",
        action="store_false",
        help="Whether to enable experiment trackers for logging.",
    )

    parser.add_argument(
            "--cache_dir",
            type=str,
            default="/scratch/sifal.klioui/cache",
            help="Where to sasve the caches files.",
        )
    
    parser.add_argument('--data_dir',
                        type=str,
                        default='/scratch/sifal.klioui/notes',
                        help='Where the training data resides.')

    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=24,
        help="The number of processes to use for the preprocessing.",
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='mosaicml/mosaic-bert-base',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--config_name",
        type=str,
        default='mosaicml/mosaic-bert-base',
        help="Pretrained config name or path if not the same as model_name",
    )

    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default='bert-base-uncased',
        help="Pretrained tokenizer name or path if not the same as model_name",
    )

    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=176,
        help="Batch size (per device) for the training dataloader.",
    )

    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=256,
        help="Batch size (per device) for the evaluation dataloader.",
    )

    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )

    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")

    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )

    # default params from mosaicBert

    """
    optimizer:
      name: decoupled_adamw
      lr: 5.0e-4 # Peak learning rate
      betas:
      - 0.9
      - 0.98
      eps: 1.0e-06
      weight_decay: 1.0e-5 # Amount of weight decay regularization
    """


    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use."
    )

    parser.add_argument('--beta1',
                        type=float,
                        default=0.9,
                        help="The value for beta1")
    parser.add_argument('--beta2',
                        type=float,
                        default=0.98,
                        help="The value for beta2")
    parser.add_argument('--eps',
                        type=float,
                        default=1e-06,
                        help="The value added to the denominator to improve numerical stability")

    parser.add_argument('--weight_decay',
                        type=float,
                        default=1e-05,
                        help="The value for weight decay coefficient ")


    """
    scheduler:
    name: linear_decay_with_warmup
    t_warmup: 0.06dur # Warmup to the full LR for 6% of the training duration
    alpha_f: 0.02 # Linearly decay to 0.02x the full LR by the end of the training duration
    """
    
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="warmup_stable_decay",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "warmup_stable_decay"],
    )

    parser.add_argument('--num_stable_steps',
                        type=int,
                        default=0,
                        help='Number of stable steps.')
    
    parser.add_argument('--num_decay_steps',
                        type=int,
                        default=None,
                        help='Number of decay steps.')
    
    parser.add_argument('--min_lr_ratio',
                        type=float,
                        default=0.02,
                        help='Minimum learning rate ratio.')
    
    parser.add_argument('--num_cycles',
                        type=float,
                        default=0.5,
                        help='Number of cycles.')
    
    parser.add_argument('--last_epoch',
                        type=int,
                        default=-1,
                        help='The index of the last epoch.')
    
    parser.add_argument('--num_warmup_steps',
                        type=int,
                        default=None,
                        help='Number of warmup steps.')

    parser.add_argument("--warmup_ratio",
                        type=float, default=0.06,
                        help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument("--max_train_steps",
                        type=int,
                        default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )

    args = parser.parse_args()
    
    if args.push_to_hub:
        if args.output_dir is None:
            raise ValueError("Need an `output_dir` to create a repo when `--push_to_hub` is passed.")

    return args


def main():
    # define MASTER_ADD & MASTER_PORT
    gpu_ids = os.environ['SLURM_JOB_GPUS'].split(",")
    os.environ['MASTER_ADDR'] = 'lisnode2'
    os.environ['MASTER_PORT'] = str(12345 + int(min(gpu_ids))) # to avoid port conflict on the same node
    os.environ['LOCAL_RANK'] = str(int(os.environ['SLURM_LOCALID']))
    os.environ['RANK'] = str(int(os.environ['SLURM_PROCID']))

    args = parse_args()
    
    accelerator_log_kwargs = {}
    
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = 'wandb'
        accelerator_log_kwargs["project_dir"] = './bert_training_logs'
    
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    
    
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
    
    
    logger.info(accelerator.state, main_process_only=False)
    
    
    
    if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    
    
    
    if accelerator.is_main_process:
            if args.push_to_hub:
                # Retrieve of infer repo_name
                repo_name = args.hub_model_id
                if repo_name is None:
                    repo_name = Path(args.output_dir).absolute().name
                # Create repo and retrieve repo_id
                api = HfApi()
                repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id
    
                with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                    if "step_*" not in gitignore:
                        gitignore.write("step_*\n")
                    if "epoch_*" not in gitignore:
                        gitignore.write("epoch_*\n")
            elif args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)
                
            os.makedirs(args.cache_dir, exist_ok = True)
        
    accelerator.wait_for_everyone()
    # this doesn't work load each one separately
    #metric = evaluate.load("precision", "accuracy", "f1", "recall")
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")

    
    raw_datasets = load_dataset(path = args.data_dir,
                                data_files={"train": os.path.join(args.data_dir,'train/*.txt'),
                                            "validation":os.path.join(args.data_dir,'validation/*.txt')},
                                cache_dir = args.cache_dir)


    config = BertConfig.from_pretrained(args.config_name, trust_remote_code=args.trust_remote_code)
    
    tokenizer = BertTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=True, trust_remote_code=args.trust_remote_code, cache_dir = args.cache_dir)

    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path, trust_remote_code = True, config = config, cache_dir = args.cache_dir)


    def tokenize_function(examples):
        
        # Remove empty lines
        examples['text'] = [
            line for line in examples['text'] if len(line) > 0 and not line.isspace()
        ]
        return tokenizer(
            examples['text'],
            padding=False,
            truncation=True,
            max_length=128, # the sequence lenght with which the model was pretrained
        
            return_special_tokens_mask=True, # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
        )

    with accelerator.main_process_first():
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=['text'],
                load_from_cache_file= not args.overwrite_cache,
                desc="Running tokenizer on dataset line_by_line",
            )
        
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    if args.debug:
        train_dataset = train_dataset.select(range(len(train_dataset)//50))
        eval_dataset = eval_dataset.select(range(len(eval_dataset)//50))

    # Conditional for small test subsets
    if len(train_dataset) > 3:
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")


    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas =(args.beta1, args.beta2), eps=args.eps, weight_decay = args.weight_decay)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.num_warmup_steps is None:
        args.num_warmup_steps = math.ceil(args.warmup_ratio * args.max_train_steps) * accelerator.num_processes
    
    if args.num_decay_steps is None:
        args.num_decay_steps = args.max_train_steps - (args.num_warmup_steps + args.num_stable_steps)


    if args.lr_scheduler_type == 'warmup_stable_decay':
        lr_scheduler = get_wsd_schedule(optimizer, args.num_warmup_steps, args.num_stable_steps, args.num_decay_steps, args.min_lr_ratio)

    """
    must handle num_warmup_steps
    else:
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps
            if overrode_max_train_steps
            else args.max_train_steps * accelerator.num_processes,
        )
    """

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)


    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("mlm_no_trainer", experiment_config)


    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0


    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break
                
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.inference_mode():
                outputs = model(**batch)

            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = prepare_sequences(accelerator.gather_for_metrics((predictions, batch['labels'])))

            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))
            accuracy.add_batch(
                predictions=predictions,
                references=references,
            )
            f1.add_batch(
                predictions=predictions,
                references=references,
            )
            precision.add_batch(
                predictions=predictions,
                references=references,
            )
            recall.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_f1 = f1.compute(average="macro")
        eval_precision = precision.compute(average="macro")
        eval_recall = recall.compute(average="macro")
        eval_accuracy = accuracy.compute()


        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(
            f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss} "
            f"eval_f1: {eval_f1} eval_precision: {eval_precision} eval_recall: {eval_recall} eval_accuracy: {eval_accuracy}")

        if args.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                    "f1": eval_f1,
                    "precision": eval_precision,
                    "recall": eval_recall,
                    "accuracy": eval_accuracy,
                },
                step=completed_steps,
            )

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, safe_serialization = False 
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                api.upload_folder(
                    commit_message=f"Training in progress epoch {epoch}",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
        
    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save,  safe_serialization = False 
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)

if __name__ == "__main__":
    main()