# Import the W&B Python Library and log into W&B
import wandb

import copy
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
from tools.lemmatization import MorphoDiTa
from tools.word_roots import DeriNet
import evaluate


from utils import *
from sqad.prompts import *


class CustomChatGemma2(BaseChatModel):
    pipeline: Any = None
    dosample: bool = False

    def __init__(self, model, tokenizer, do_sample=False, 
                 max_new_tokens=10, precision="fp16", device_map="auto", **kwargs):
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
        self.all_scores = []

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        testing_dataset = load_from_disk("czech-bench/benchmarks/sqad/data/test")
        testing_dataset = testing_dataset.shuffle(seed=228).select(range(100))

        tokenizer =  AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        llm = CustomChatGemma2(None, tokenizer, do_sample=False, max_new_tokens=10)
        llm.pipeline.model = self.trainer.model

        lemmatizer = MorphoDiTa()
        if not lemmatizer.lemmatizer:
            lemmatizer = False
        root_lexicon = None
        if not lemmatizer:
            print("Failed to load lemmatizer model, morphological analysis will be skipped")
        else:
            try:
                root_lexicon = DeriNet()
            except:
                print("Failed to load root lexicon, root extraction will be skipped")

        prompt = PROMPT_SELECTOR.get_prompt(llm)
        str_parser = StrOutputParser()

        print("Evaluating SQAD")

        prompt = PROMPT_SELECTOR.get_prompt(llm)
        str_parser = StrOutputParser()

        def morpho_analyze(answer):
            tokens = [tok.rstrip(",.?!") for tok in answer.lower().rstrip('\r\n').split()]
            lemmas = ""
            roots = ""
            for token in tokens:
                lemma = lemmatizer.lemmatize(token)
                lemmas += lemma + " "      
                if root_lexicon is not None:
                    roots += root_lexicon.get_root(lemma) + " "
            return lemmas, roots

        references = []
        predictions = []
        ref_lemmas = []
        pred_lemmas = []
        ref_roots = []
        pred_roots = []
        count = 0
        cum_time = 0.

        for i, example in enumerate(testing_dataset):
            print(f"\rExample {i+1} / {len(testing_dataset)}", end="")

            if i > 99:
                break

            context = example["context"]
            question = example["question"]

            try:
                start_time = time.time()
                if is_chat_model(llm):
                    result = llm.invoke(prompt.format_prompt(context=context, question=question).to_messages())
                else:
                    result = llm.invoke(prompt.format_prompt(context=context, question=question).text)
                result = str_parser.invoke(result)
                end_time = time.time()
            except Exception as e:
                print(f"\nExample skipped due to an LLM Error: {e}")
                continue
            
            id = example["item_id"]
            ref = {
                "answers": {"text": example["answers"], "answer_start": [d["start"] for d in example["occurrences"]]},
                "id": id
            }
            ans = {
                "prediction_text": result,
                "id": id
            }
            references.append(ref)
            predictions.append(ans)

            if lemmatizer:
                ans_lemmas = []
                ans_roots = []
                for ans_text in example["answers"]:
                    lems, roots = morpho_analyze(ans_text)
                    ans_lemmas.append(lems)
                    ans_roots.append(roots)

                ref_lem_dict = copy.deepcopy(ref)
                ref_lem_dict["answers"]["text"] = ans_lemmas
                ref_root_dict = copy.deepcopy(ref)
                ref_root_dict["answers"]["text"] = ans_roots

                lems, roots = morpho_analyze(result)
                pred_lem_dict = copy.deepcopy(ans)
                pred_lem_dict["prediction_text"] = lems
                pred_root_dict = copy.deepcopy(ans)
                pred_root_dict["prediction_text"] = roots

                ref_lemmas.append(ref_lem_dict)
                pred_lemmas.append(pred_lem_dict)
                if root_lexicon is not None:
                    ref_roots.append(ref_root_dict)
                    pred_roots.append(pred_root_dict)
            count += 1
            cum_time += end_time - start_time
            
        print("\nComputing metrics")

        exact_match = 0
        bow_f1 = 0
        lemmas_exact_match = 0
        lemmas_bow_f1 = 0
        roots_exact_match = 0
        roots_bow_f1 = 0
        average_inference_time = 0
        total_valid_examples = 0

        lines = "\nResults:\n"
        if count > 0:
            metric = evaluate.load("squad")
            res = metric.compute(predictions=predictions, references=references)
            lines += f"Exact Match: {res['exact_match']:.2f}\n"
            lines += f"BoW F1: {res['f1']:.2f}\n"
            exact_match = res['exact_match']
            bow_f1 = res['f1']

            if lemmatizer:
                res_lemma = metric.compute(predictions=pred_lemmas, references=ref_lemmas)
                lines += f"Lemmas Exact Match: {res_lemma['exact_match']:.2f}\n"
                lines += f"Lemmas BoW F1: {res_lemma['f1']:.2f}\n"
                lemmas_exact_match = res_lemma['exact_match']
                lemmas_bow_f1 = res_lemma['f1']

                if root_lexicon is not None:
                    res_root = metric.compute(predictions=pred_roots, references=ref_roots)
                    lines += f"Roots Exact Match: {res_root['exact_match']:.2f}\n"
                    lines += f"Roots BoW F1: {res_root['f1']:.2f}\n"
                    roots_exact_match = res_root['exact_match']
                    roots_bow_f1 = res_root['f1']
            
            lines += f"Average inference time: {cum_time/count:.2f}s\n"
            average_inference_time = cum_time/count

        lines += f"Total valid examples used: {count}\n"
        total_valid_examples = count

        self.all_scores.append(exact_match)
        print("All scores:", self.all_scores)

        # save metrics to SQAD/
        self._wandb.log({"SQAD/exact_match": exact_match, "SQAD/bow_f1": bow_f1, 
                         "SQAD/lemmas_exact_match": lemmas_exact_match, "SQAD/lemmas_bow_f1": lemmas_bow_f1,
                         "SQAD/roots_exact_match": roots_exact_match, "SQAD/roots_bow_f1": roots_bow_f1,
                         "SQAD/average_inference_time": average_inference_time, "SQAD/total_valid_examples": total_valid_examples})
        
        # self._wandb.log({"agree_accuracy": accuracy, "parse_fails": parse_fails})


# 1: Define objective/training function
def objective(config):

    # run training with config
    cfg = DotDict(load_config("czech-llm-finetuning/config/arc_no_expl.yaml"))

    seed = cfg.hyperparameters.seed
    random.seed(seed)
    model_name = "google/gemma-2-2b-it"
    
    output_dir_name = cfg.hyperparameters.output_dir + "r="+str(config.r)+"_alpha="+str(config.alpha)+"_lr="+str(cfg.hyperparameters.learning_rate)

    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='eager')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.padding_side = "left"

    datasets = load_from_disk(cfg.dataset)

    def tokenize_function(examples):
        # texts = list(map(lambda x,y: x + y, examples["input"], examples["output"]))
        texts = list(map(lambda x: x, examples["text"]))
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

    print("Validation dataset size:", len(validation_dataset))
    for i in range(40):
        print(validation_dataset[i]["text"])
        print()

    lora_config = LoraConfig(
        r=config.r,
        lora_alpha=config.alpha,
        target_modules=cfg.hyperparameters.lora_config.target_modules,
        lora_dropout=cfg.hyperparameters.lora_config.dropout,
        bias=cfg.hyperparameters.lora_config.bias,
        task_type=cfg.hyperparameters.lora_config.task_type,
    )

    model.enable_input_require_grads()

    model = get_peft_model(model, lora_config)

    # model = activate_adapter(model, "training/lora/cp/instruction_gemma2-2b-it")

    model.print_trainable_parameters()

    run_name = cfg.hyperparameters.run_name + "_r=" + str(config.r) + "_alpha=" + str(config.alpha) + "_: " + get_readable_datetime_string()
    
    wandb.init(entity=cfg.wandb.entity,
               project=cfg.wandb.project,
               config=dict(cfg),
               job_type="training",
               name=run_name
            )

    training_args = TrainingArguments(
        run_name=run_name,
        output_dir=output_dir_name,  # The output directory
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
    
    # Instantiate the WandbPredictionProgressCallback
    progress_callback = WandbPredictionProgressCallback(
        trainer=trainer,
        tokenizer=tokenizer,
        val_dataset=validation_dataset,
        freq=2,
    )

    # Add the callback to the trainer
    # trainer.add_callback(progress_callback)

    trainer.train()
    model.save_pretrained(output_dir_name)

    best_score = max(progress_callback.all_scores) if progress_callback.all_scores else 0

    print("Best score:", best_score)

    del model
    del trainer
    torch.cuda.empty_cache()

    return best_score

def main():
    wandb.init(project="czech-llm")
    sqad_em = objective(wandb.config)
    wandb.log({"sqad_em": sqad_em})

# 2: Define the search space
sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "minimize", "name": "sqad_em"},
    "parameters": {
        "r": {"values": [256, 512]},
        "alpha": {"values": [256, 512]},
        # "lr": {"values": [0.00005]},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="czech-llm")

wandb.agent(sweep_id, function=main)
