import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

from transformers.integrations import WandbCallback
import pandas as pd

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from transformers import pipeline, QuantoConfig, AutoTokenizer, AutoModelForCausalLM
import torch
from peft import LoraConfig, get_peft_model, PeftModel

import argparse
import random
import wandb
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from utils import *
from agree.prompts import *


class CustomChatGemma2(BaseChatModel):
    pipeline: Any = None
    dosample: bool = False

    def __init__(self, model, tokenizer, do_sample=False, 
                 max_new_tokens=5, precision="fp16", device_map="auto", **kwargs):
        super().__init__()

        model=AutoModelForCausalLM.from_pretrained(
                "google/gemma-2-2b-it",
                device_map=device_map,
                torch_dtype=torch.float16
            )

        self.pipeline = pipeline(
            "text-generation",
            model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            model_kwargs=kwargs,
            device_map=device_map,
            )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        
        system = messages[0].content
        human = messages[1].content

        messages = [
            # {"role": "system", "content": system},
            {"role": "user", "content": system + '\n' + human},
            # {"role": "user", "content": human},
        ]

        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        ]

        # print("\n----")

        outputs = self.pipeline(
            prompt,
            eos_token_id=terminators,
            do_sample=self.dosample,
        )
        
        result = outputs[0]["generated_text"][len(prompt):]
        # print(result)

        message = AIMessage(content=result)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "gemma-2-2b-it_lora"
    

class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model's accuracy."""

    def __init__(self, trainer, tokenizer, val_dataset, freq=2):
        """Initializes the WandbPredictionProgressCallback instance."""
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # control the frequency of logging by logging the predictions
        # every `freq` epochs
        if state.epoch % self.freq == 0 or True:
            # generate predictions
            # predictions = self.trainer.model.generate()
            # decode predictions and labels
            # predictions = decode_predictions(self.tokenizer, predictions)
            # add predictions to a wandb.Table

            # messages = [
            #     {"role": "user", "content": "Ahoj, jak se máš?"},
            # ]

            tokenizer =  AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
            llm = CustomChatGemma2(None, tokenizer, do_sample=True)
            llm.pipeline.model = self.trainer.model

            sys_message = SystemMessage(content="Jsi assistent.", type="system")
            message = BaseMessage(content="Ahoj, jak se máš?", type="text")

            print(llm.invoke([sys_message, message]).content)

            testing_dataset = json.load(open("czech-bench/benchmarks/agree/data/test.json", "r"))

            prompt = PROMPT_SELECTOR.get_prompt(llm)
            str_parser = StrOutputParser()

            correct = 0
            parse_fails = 0
            count = 0
            cum_time = 0.

            print(f"{BLUE}Evaluating AGREE\n{RESET}")

            for i, example in tqdm(enumerate(testing_dataset)):
                # if i+1 > 10:
                #     break
                # print(f"\rExample {i+1} / {len(testing_dataset)}", end="")
                sentence = example["sentence"]
                choices = example["choices"]
                gt = example["answer_idx"] + 1

                try:
                    start_time = time.time()
                    if is_chat_model(llm):
                        result = llm.invoke(prompt.format_prompt(sentence=sentence, choices=choices).to_messages())
                    else:
                        result = llm.invoke(prompt.format_prompt(sentence=sentence, choices=choices).text)    
                    result = str_parser.invoke(result)
                    end_time = time.time()
                    res_split = result.split()
                    if res_split:
                        res = result.split()[0].strip().strip(")")
                    else:
                        res = result
                except Exception as e:
                    print(f"\nExample skipped due to an LLM Error: {e}")
                    continue
                
                try:
                    prediction = int(res)
                except:
                    parse_fails += 1
                    continue
                if prediction == gt:
                    correct += 1
                count += 1
                cum_time += end_time - start_time

            print("\nComputing metrics")

            accuracy = correct/count*100
            total_valid_examples = count
            print(f"Accuracy: {accuracy:.2f}")
            print(f"Unpareable answersz: {parse_fails}")
            print("Total valid examples used: ", total_valid_examples)

            self._wandb.log({"agree_accuracy": accuracy, "parse_fails": parse_fails})



def decode_predictions(tokenizer, predictions):
    labels = tokenizer.batch_decode(predictions.label_ids)
    logits = predictions.predictions.argmax(axis=-1)
    prediction_text = tokenizer.batch_decode(logits)
    return {"labels": labels, "predictions": prediction_text}


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

    model = get_peft_model(model, lora_config)

    # model = activate_adapter(model, "training/lora/cp/instruction_gemma2-2b-it")

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

    # # First, instantiate the Trainer
    # trainer = Trainer(
        # model=model,
        # args=training_args,
        # train_dataset=lm_datasets["train"],
        # eval_dataset=lm_datasets["validation"],
    # )

    trainer = Trainer(model=model, args=training_args, 
                      train_dataset=train_dataset, 
                      eval_dataset=validation_dataset,
                      )
    
    # Instantiate the WandbPredictionProgressCallback
    progress_callback = WandbPredictionProgressCallback(
        trainer=trainer,
        tokenizer=tokenizer,
        val_dataset=validation_dataset,
        freq=2,
    )

    # Add the callback to the trainer
    trainer.add_callback(progress_callback)


    trainer.train()
    model.save_pretrained(cfg.hyperparameters.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="czech-llm-finetuning/config/instruction_gemma2_config.yaml", help="Path to the config file.")
    main(parser.parse_args().config)


