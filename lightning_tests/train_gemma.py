import torch
import litgpt
from litgpt.model import GPT
from litgpt.data import Alpaca2k
import lightning as L
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

L.seed_everything(228)


class GemmaLit(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", attn_implementation='eager')
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
        self.model.train()
        # for param in self.model.parameters():
        #     param.requires_grad = True

    def training_step(self, batch):
        # print(f"\033[91mModel is in {'train' if self.model.training else 'eval'} mode\033[0m")

        # Send the batch through the model and calculate the loss
        # The Trainer will run .backward(), optimizer.step(), .zero_grad(), etc. for you
        input_ids, targets = batch["input_ids"], batch["labels"]
        logits = self.model(input_ids) # gets CausalLMOutputWithPast object, 'CausalLMOutputWithPast' object has no attribute 'argmax'
        # print logits with color:
        # print("\033[91m" + self.tokenizer.decode(logits.logits.argmax(-1)[0]) + "\033[0m")
        loss = litgpt.utils.chunked_cross_entropy(logits.logits[..., :-1, :], targets[..., 1:])
        # print("\033[92m" + self.tokenizer.decode(targets[0]) + "\033[0m")
        self.log("train_loss", loss, prog_bar=True)
        return loss


    def configure_optimizers(self):
        # Choose an optimizer or implement your own.
        return torch.optim.AdamW(self.model.parameters(), lr=0.0002, weight_decay=0.0, betas=(0.9, 0.95))


if __name__ == "__main__":
    data = Alpaca2k()
    tokenizer = litgpt.Tokenizer("checkpoints/google/gemma-2-2b/")
    data.connect(tokenizer, batch_size=1, max_seq_length=512)

    # with trainer.init_module():
    #     model = GemmaLit()
        # print(model)
    model = GemmaLit()

    trainer = L.Trainer(
        devices=1,
        max_epochs=1,
        max_steps=50,
        accumulate_grad_batches=2,
        precision="bf16-true",
        accelerator="cuda",
        strategy="ddp"
    )
    trainer.fit(model, data)

    # Save final checkpoint
    trainer.save_checkpoint("checkpoints/google/finetuned_gemma-2-2b.ckpt", weights_only=True)

    trainer.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

