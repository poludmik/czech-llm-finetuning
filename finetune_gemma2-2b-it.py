import argparse
import random
import wandb
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

from utils import *


def main(config_path: str = "config.yaml"):
    cfg = DotDict(load_config(config_path))

    seed = cfg.hyperparameters.seed
    random.seed(seed)
    model_name = "google/gemma-2-2b-it"

    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='eager')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.padding_side = "left"

    datasets = load_from_disk(cfg.dataset)

    def tokenize_function(examples):
        texts = list(map(lambda x,y: x + y, examples["input"], examples["output"]))
        # toks = tokenizer(texts, truncation=True, max_length=cfg.hyperparameters.trainer.max_seq_length, padding='max_length')
        # print(">>>" + texts[0] + "<<<")
        toks = tokenizer(texts)
        return toks

    datasets = datasets.map(tokenize_function, batched=True)
    datasets = datasets.map(lambda x: {'labels':x['input_ids']})

    num_validation_samples = int(len(datasets) * cfg.hyperparameters.split_ratio)
    split_dataset = datasets.train_test_split(test_size=num_validation_samples, seed=seed)

    train_dataset = split_dataset['train']
    validation_dataset = split_dataset['test']

    lora_config = LoraConfig(
        r=cfg.hyperparameters.lora_config.r,
        lora_alpha=cfg.hyperparameters.lora_config.alpha,
        target_modules=cfg.hyperparameters.lora_config.target_modules,
        lora_dropout=cfg.hyperparameters.lora_config.dropout,
        bias=cfg.hyperparameters.lora_config.bias,
        task_type=cfg.hyperparameters.lora_config.task_type,
    )

    model.enable_input_require_grads()

    # model = get_peft_model(model, lora_config)

    model = activate_adapter(model, "training/lora/instruction_gemma2-2b-it")

    model.print_trainable_parameters()

    run_name = cfg.hyperparameters.run_name + ": " + get_readable_datetime_string()
    
    wandb.init(entity=cfg.wandb.entity,
               project=cfg.wandb.project,
               config=dict(cfg),
               job_type="training",
               name=run_name
            )

    training_args = TrainingArguments(
        run_name=run_name,
        output_dir=cfg.hyperparameters.output_dir,  # The output directory
        overwrite_output_dir=cfg.hyperparameters.overwrite_output_dir,  # overwrite the content of the output directory
        evaluation_strategy=cfg.hyperparameters.evaluation_strategy,
        max_steps=cfg.hyperparameters.max_steps, # instead of epochs
        per_device_train_batch_size=cfg.hyperparameters.per_device_train_batch_size,  # batch size for training
        per_device_eval_batch_size=cfg.hyperparameters.per_device_eval_batch_size,  # batch size for evaluation
        gradient_accumulation_steps=cfg.hyperparameters.gradient_accumulation_steps,
        eval_accumulation_steps=cfg.hyperparameters.eval_accumulation_steps,
        save_steps=cfg.hyperparameters.save_steps,
        learning_rate=cfg.hyperparameters.learning_rate,
        logging_steps=cfg.hyperparameters.logging_steps, # how often to log to W&B
        eval_steps=cfg.hyperparameters.eval_steps,  # Number of update steps between two evaluations.
        warmup_ratio=cfg.hyperparameters.warmup_ratio,  # number of warmup steps for learning rate scheduler
        # lr_scheduler_kwargs={"min_lr_ratio": cfg.hyperparameters.min_lr_ratio},
        weight_decay = cfg.hyperparameters.weight_decay,
        fp16=cfg.hyperparameters.fp16,
        lr_scheduler_type = cfg.hyperparameters.lr_scheduler_type,
        report_to="wandb",
        gradient_checkpointing=cfg.hyperparameters.gradient_checkpointing,
        # neftune_noise_alpha=cfg.hyperparameters.neftune_noise_alpha,
        )

    trainer = Trainer(model=model, args=training_args, 
                      train_dataset=train_dataset, 
                      eval_dataset=validation_dataset,
                      )

    trainer.train()
    model.save_pretrained(cfg.hyperparameters.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="czech-llm-finetuning/config/instruction_gemma2_config.yaml", help="Path to the config file.")
    main(parser.parse_args().config)