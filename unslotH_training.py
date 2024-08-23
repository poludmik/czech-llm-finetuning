from unsloth import FastLanguageModel
from utils import *
# import torch
max_seq_length = 3500 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-2-2b-it",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 228,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


from datasets import load_dataset, load_from_disk
dataset = load_from_disk("dataset-playground/small_datasets/requests_translated_0_to_10k_google_gemma-2-2b-it")

def tokenize_function(examples):
    texts = list(map(lambda x,y: x + y, examples["input"], examples["output"]))
    # toks = tokenizer(texts, truncation=True, max_length=cfg.hyperparameters.trainer.max_seq_length, padding='max_length')
    # print(">>>" + texts[0] + "<<<")
    return {"text": texts}

dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.map(lambda x: {'labels': x['input']})

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import wandb

wandb.init(entity="poludmik", project="czech-llm", 
            name = "Unsloth_test" + ": " + get_readable_datetime_string(),
            config={"learning_rate": 0.000025, "max_steps": 15000})

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 1,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        report_to="wandb",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,
        warmup_steps = 500,
        max_steps = 15000,
        save_steps=1000,
        learning_rate = 0.000025,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 25,
        optim = "adamw_8bit", # paged_adamw_32bit
        weight_decay = 0.1,
        lr_scheduler_type = "cosine",
        seed = 228,
        output_dir = "training/unsloth/",
    ),
)

trainer_stats = trainer.train()

wandb.finish()
