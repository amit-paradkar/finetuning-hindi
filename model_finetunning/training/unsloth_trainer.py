from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from peft import LoraConfig

class UnslothTrainer:
    def __init__(self, config_path: str, pretrained_model_path: str = None):
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.pretrained_model_path = pretrained_model_path
        self.load_model()
        self.prepare_dataset()
        self.build_trainer()

    def load_model(self):
        model_path = self.pretrained_model_path or self.config["model_name"]

        self.tokenizer, self.model, _ = FastLanguageModel.from_pretrained(
            model_name=model_path,
            quantization_config={"load_in_4bit": self.config.get("load_in_4bit", False)},
            device_map="auto"
        )

        self.tokenizer.model_max_length = self.config["dataset"]["max_seq_length"]

        if self.config.get("use_peft", True):
            peft_cfg = self.config["peft_config"]
            peft_config = LoraConfig(
                r=peft_cfg["r"],
                lora_alpha=peft_cfg["lora_alpha"],
                lora_dropout=peft_cfg["lora_dropout"],
                bias=peft_cfg["bias"],
                task_type=peft_cfg["task_type"],
                target_modules=peft_cfg["target_modules"]
            )
            self.model = FastLanguageModel.get_peft_model(self.model, peft_config)

    def prepare_dataset(self):
        dataset_path = self.config["dataset"]["path"]
        dataset_type = self.config["dataset"]["type"]
        dataset_format = self.config["dataset"].get("format", None)

        # Load from HF hub or local files
        if dataset_format in ["json", "parquet", "csv"]:
            dataset = load_dataset(dataset_format, data_files=dataset_path)
        else:
            # Assume it's a Hugging Face dataset package
            dataset = load_dataset(dataset_path)

        if dataset_type == "chat":
            input_map_fn = self.chat_map_fn
        elif dataset_type == "wiki":
            input_map_fn = self.wiki_map_fn
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        self.dataset = FastLanguageModel.prepare_dataloaders(
            tokenizer=self.tokenizer,
            dataset=dataset["train"],  # or use self.config["dataset"].get("split", "train")
            max_seq_length=self.config["dataset"]["max_seq_length"],
            batch_size=self.config["training"]["batch_size"],
            packing=True,
            shuffle=True,
            input_map_fn=input_map_fn
        )

    def build_trainer(self):
        training_args = TrainingArguments(
            output_dir=self.config["training"]["output_dir"],
            num_train_epochs=self.config["training"]["epochs"],
            per_device_train_batch_size=self.config["training"]["batch_size"],
            gradient_accumulation_steps=self.config["training"]["grad_accumulation"],
            learning_rate=self.config["training"]["lr"],
            save_steps=self.config["training"]["save_steps"],
            logging_steps=self.config["training"]["logging_steps"],
            evaluation_strategy="no",
            report_to="none",
            bf16=self.config["training"].get("bf16", False),
            fp16=self.config["training"].get("fp16", False),
        )

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=self.dataset,
        )

    def chat_map_fn(self, sample):
        messages = sample["conversations"]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": prompt}

    def wiki_map_fn(self, sample):
        return {"text": f"<|title|>{sample['title']}\n<|text|>{sample['text']}"}

    def train(self):
        self.trainer.train()
