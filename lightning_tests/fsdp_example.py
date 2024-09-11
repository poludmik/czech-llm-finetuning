
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.demos import Transformer, WikiText2
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.strategies import FSDPStrategy


from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

import pytorch_lightning as pl

# intialize model, optimizer and defines training step
class LanguageModel(pl.LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", attn_implementation='eager')
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
        self.model.train()

    def training_step(self, batch):
        input, target = batch
        logits = self.model(input)
        # print(target.shape)
        # loss = F.nll_loss(output.logits, target)

        # Reshape logits and targets for the loss function
        logits = logits.logits.view(-1, logits.logits.size(-1))  # Shape: [batch_size * seq_len, vocab_size]
        target = target.view(-1)  # Shape: [batch_size * seq_len]
        
        # Calculate loss using cross-entropy
        loss = F.cross_entropy(logits, target)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.1)

pl.seed_everything(42)

# Data
dataset = WikiText2()
train_dataloader = DataLoader(dataset, batch_size=64)
print("Dataloader created:", train_dataloader)
print("Dataloader length:", len(train_dataloader))
print("Dataloader batch size:", train_dataloader.batch_size)

print("get item from dataloader:", next(iter(train_dataloader))[0].shape)

# Model
model = LanguageModel(vocab_size=dataset.vocab_size)

layers = {
        nn.TransformerEncoderLayer,
        nn.TransformerDecoderLayer,
    }
strategy = FSDPStrategy(
    auto_wrap_policy=layers,
    activation_checkpointing_policy=layers,
    cpu_offload=True
)

# Trainer
trainer = pl.Trainer(accelerator="cuda", devices=1, strategy=strategy)
trainer.fit(model, train_dataloader)
trainer.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
