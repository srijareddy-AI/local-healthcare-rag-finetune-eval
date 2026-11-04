"""Fine-tuning for structured extraction (LoRA, local, zero paid APIs).

This is REAL fine-tuning, not prompting: it trains low-rank adapter
weights on the synthetic discharge notes in data/discharge_notes/notes.json
(train split) so the model learns to emit the fixed extraction schema.

Base model: distilgpt2 (82M parameters, CPU-friendly). Only LoRA adapter
parameters are trained; the base model stays frozen. Requires:
    pip install -r requirements-finetune.txt
and a Hugging Face model download on first run (public weights, no API key).

Usage:
    python src/finetune.py --epochs 3 --output models/lora-extraction
    python src/finetune.py --smoke-test   # 1 tiny epoch, proves the loop runs

If training cannot run on a given machine (RAM, download, or time), this
script still documents the exact procedure — do not report metrics from a
run that did not happen. See README.md, "What ran, what did not".
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build_examples():
    notes = json.loads((ROOT / "data" / "discharge_notes" / "notes.json").read_text())
    examples = []
    for item in notes:
        if item["split"] != "train":
            continue
        prompt = (
            "Extract structured data from this SYNTHETIC discharge note. "
            "Return only a JSON object with keys medications, follow_up_days, "
            "follow_up_clinic, red_flags, contact. If a field is not present, "
            "use null or [].\n\n"
            f"Note: {item['note']}\n\nJSON:"
        )
        examples.append({"prompt": prompt, "completion": json.dumps(item["gold"])})
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--output", default="models/lora-extraction")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        args.epochs = 1

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    examples = build_examples()
    if args.smoke_test:
        examples = examples[:2]
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    def tokenize(example):
        text = example["prompt"] + " " + example["completion"] + tokenizer.eos_token
        tokens = tokenizer(text, truncation=True, max_length=512)
        tokens["labels"] = list(tokens["input_ids"])
        return tokens

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = [tokenize(r) for r in rows]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return self.rows[i]

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=Dataset(examples))
    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved LoRA adapter to {args.output}")


if __name__ == "__main__":
    main()
