from unsloth_trainer import UnslothTrainer

# === Stage 1: Wiki full fine-tune ===
wiki_trainer = UnslothTrainer(config_path="configs/dry_run/wiki.yaml")
wiki_trainer.train()
wiki_trainer.model.save_pretrained("checkpoints/wiki_model")
wiki_trainer.tokenizer.save_pretrained("checkpoints/wiki_model")

# === Stage 2: Alpaca LoRA fine-tune ===
alpaca_trainer = UnslothTrainer(config_path="configs/dry_run/alpaca.yaml")
alpaca_trainer.train()


# Push to hub
#trainer.push_to_hub()

