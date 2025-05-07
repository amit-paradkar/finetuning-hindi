from transformers import TrainerCallback
import os
import shutil

class PushToHubCallback(TrainerCallback):
    def __init__(self, trainer, repo_name, push_interval_steps=500, keep_last_n=2, private=True):
        self.trainer = trainer
        self.repo_name = repo_name
        self.push_interval_steps = push_interval_steps
        self.keep_last_n = keep_last_n
        self.private = private

    def on_step_end(self, args, state, control, **kwargs):
        """Called at every step end."""
        if state.global_step % self.push_interval_steps == 0 and state.global_step != 0:
            print(f"🚀 Pushing checkpoint at step {state.global_step} to HuggingFace Hub...")

            # Save model locally first
            self.trainer.save_model()

            # Push model + tokenizer to hub
            self.trainer.model.push_to_hub(self.repo_name, private=self.private)
            self.trainer.tokenizer.push_to_hub(self.repo_name, private=self.private)

            # Clean old checkpoints
            self._cleanup_checkpoints(args.output_dir)

    def _cleanup_checkpoints(self, checkpoint_dir):
        """Keeps only the last N checkpoints."""
        checkpoints = [ckpt for ckpt in os.listdir(checkpoint_dir) if ckpt.startswith("checkpoint-")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        if len(checkpoints) > self.keep_last_n:
            to_delete = checkpoints[:-self.keep_last_n]
            for ckpt in to_delete:
                full_path = os.path.join(checkpoint_dir, ckpt)
                print(f"🗑️ Deleting old checkpoint: {full_path}")
                shutil.rmtree(full_path)
