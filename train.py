from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch

class Trainer:
    def __init__(self, model_name, Data):
        self.model_name = model_name
        self.dtype = torch.bfloat16
        self.dataset = Data

    def format_text_generation(self, example):
        question = example['question'] if example['question'] is not None else ""
        context = example['context'] if example['context'] is not None else ""
        answers = example['answers'] if example['answers'] is not None else {"text": [""]}
        answer_text = answers["text"][0] if answers["text"] else ""
        example['text'] = f"Question: {question}\nContext: {context}\nAnswer: {answer_text}"
        return example

    def map_dataset(self):
        dataset = self.dataset.map(self.format_text_generation)
        dataset = dataset.remove_columns(['id', 'title', 'context', 'question', 'answers'])
        dataset = dataset.filter(lambda example: example['text'].strip() != "")
        return dataset

    def load_model(self, model_name, dtype, bnb_config):
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, torch_dtype=dtype # Use torch_dtype instead of dtype
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return {"model": model, "tokenizer": tokenizer}

    def hyper_config(self):
        model_name = self.model_name
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.dtype,
            bnb_4bit_use_double_quant=False,
        )
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["c_attn", "c_proj", "c_fc"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )

        llm = self.load_model(model_name, self.dtype, bnb_config)
        model = llm["model"]
        tokenizer = llm["tokenizer"]

        # Ensure trainable parameters have gradients enabled and are in the correct dtype
        model = get_peft_model(model, lora_config)
        for param in model.parameters():
            if param.ndim == 1:
                # cast layer norm to float32 for stability
                param.data = param.data.to(torch.float32)

        model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir="output",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_steps=1000,
            logging_steps=2,
            remove_unused_columns=True,
            optim="paged_adamw_8bit" # Ensure using 8bit optimizer
        )

        return({"model": model, "tokenizer": tokenizer, "training_args": training_args, "lora_config": lora_config})

    def main_train(self):
        config = self.hyper_config()
        dataset = self.map_dataset()

        model = config["model"]
        tokenizer = config["tokenizer"]
        training_args = config["training_args"]
        lora_config = config["lora_config"] # Get lora_config from the dictionary

        # Set padding token for the model
        model.config.pad_token_id = tokenizer.pad_token_id

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
            peft_config=lora_config, # Pass lora_config to SFTTrainer
            max_seq_length=512 # Ensure max_seq_length is set
        )

        trainer.train()

dataset = load_dataset("squad", split="train")
train = Trainer("distilgpt2", dataset)
train.main_train()
