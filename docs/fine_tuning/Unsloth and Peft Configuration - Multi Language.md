# Multi lingual finetuning -  Parameters for Unsloth and PEFT

## What is Unsloth

The Unsloth library is a new framework (2024) that focuses on making fine-tuning and pretraining of large language models (LLMs) super memory- and compute-efficient. It’s mainly famous for:

- Allowing very long context lengths (like 8k, 32k, even 128k tokens).
- Training big models on small hardware (even 24GB VRAM GPUs).
- Being crazy fast while still producing high-quality models.

Originally, Unsloth was mainly for fine-tuning (e.g., LoRA, QLoRA), but it also supports pretraining now — that's the part you're asking about.

## How Unsloth Works for Pretraining:
1. Efficient Model Rewrites

    Unsloth rewrites the core model operations (like attention, feedforward, normalization) to use much more memory-efficient and faster CUDA kernels.

    Instead of normal PyTorch ops, it uses custom kernels designed for long-sequence optimization.
    It also fuses operations (e.g., fused RMSNorm, fused attention) to reduce memory and computation time.
    Why it matters for pretraining:
    You can train bigger models with longer sequences without hitting memory walls.

2. Memory Optimization Tricks

    <u>Gradient checkpointing:</u>
    Instead of storing every intermediate activation, it recomputes them during backward pass — saving RAM.
    
    <u>Paged Attention:</u>
    Inspired by Meta's "Paged Attention" (Llama 3), Unsloth uses a system that dynamically loads only the needed parts of attention memory, so you can scale to huge sequence lengths like 128k tokens.
    
    <u>Quantization-aware Pretraining:</u>
    You can pretrain models with some layers in 4-bit or 8-bit, which slashes memory use even more (without hurting performance too much).

3. Tokenizer and Data Loader Improvements

    Unsloth uses ultra-fast tokenizers that are optimized for huge datasets (e.g., text pretraining data).
    The data loading pipeline is heavily parallelized, shuffling, batching, and tokenizing data super fast to keep GPUs fully utilized.
4. Flash Attention v2 and v3 Support

    For attention mechanism, Unsloth uses Flash Attention v2 or v3 — much faster and lower memory usage than vanilla attention.
    Flash Attention is critical for handling long context lengths during pretraining.

5. Trainer API for Pretraining

Unsloth gives you a high-level API that makes pretraining easy:
```
from unsloth import FastPreTrainer

trainer = FastPreTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_data,
    config = {
        "max_seq_length": 8192,
        "batch_size": 8,
        "learning_rate": 2e-4,
        ...
    }
)
trainer.train()
```
It handles:

- Optimizer setup (e.g., AdamW with special weight decay)
- Learning rate schedules
- Distributed training (even on multiple GPUs or nodes)
- Automatic mixed precision (FP16 / BF16)

So in short:

✅ Model optimized → Faster and less memory.

✅ Data pipeline optimized → Keeps GPUs busy.

✅ Attention optimized → Handles crazy-long sequences.

✅ Trainer API → Easy to start pretraining.

✅ Pretraining-specific improvements → Grad checkpointing, quantized layers, longer context.

## Unsloth Training parameters
Language morphology and dataset availability are the two factors on which training parameters are decided. Below tables shows examples of the classification

| Language | Mophologically complex | Corpus |
| ------   | -----------------------| -------------- |
| English  |  Simple                | High           |
| ------   | -----------------------| -------------- |
| Spanish  |  Simple                | High           |
| ------   | -----------------------| -------------- |
| French   |  Simple                | High           |
| ------   | -----------------------| -------------- |
|Mandarin  |  Complex       | High           |
| ------   | -----------------------| -------------- |
| Japanese  |  Complex                | High           |
| ------   | -----------------------| -------------- |
| Korean   |  Complexe                | High           |
| ------   | -----------------------| -------------- |
| Arabic  |  Complex                | High           |
| ------   | -----------------------| -------------- |
| Hindi    |  Complexe                | High           |
| ------   | -----------------------| -------------- |
| Marathi  |  Complex                | High           |
| ------   | -----------------------| -------------- |



Based on the target language the following Unsloth parameters are tweaked:

→ Different sequence lengths

→ Different tokenization patterns

→ Different batch sizes needed

→ Different optimizer tweaks sometimes


<u>Below is detailed list and explanation:</u>
1. <u>max_seq_length</u>

    🔹 Insight:

    Languages like Chinese, Japanese, Korean (CJK languages) are token-dense — one token ≈ one character.

    Languages like English, Spanish, French are token-light — a single token can be multiple letters.

    🔹 Adjustment:

    For CJK, you might want shorter max_seq_length (like 2048 or 4096), because you'll hit context windows faster.

    For token-light languages, longer max_seq_length (like 4096–8192) makes sense.

    ```
    max_seq_length = 4096  # English
    max_seq_length = 2048  # Chinese
    ```

2. <u>batch_size</u>

    🔹 Insight:

    High token density = each sample is bigger (in bytes).

    GPU memory will fill up faster for "heavy" languages (CJK, Arabic).

    🔹 Adjustment:

    Use smaller batch sizes for heavy/token-dense languages.
    Use larger batch sizes for lighter languages.

    ```
    batch_size = 16  # Spanish
    batch_size = 8   # Japanese

    ```

3. learning_rate

    🔹 Insight:

    Low-resource languages (e.g., Swahili, Uzbek) = risk of overfitting fast.

    High-resource languages (English, Chinese) = can afford higher learning rates.

    🔹 Adjustment:

    If fine-tuning on a small dataset or a rare language, use a lower learning rate to avoid catastrophic forgetting.

    ```
    learning_rate = 2e-5  # Urdu fine-tuning (small corpus)
    learning_rate = 2e-4  # English fine-tuning (huge corpus)

    ```
4. optimizer settings

    🔹 Insight:

    Different language datasets have different "difficulty" levels (typos, grammar complexity).
    Languages with rich morphology (Finnish, Turkish) sometimes benefit from slight optimizer tweaks.
    
    🔹 Adjustment:

    Use optimizers like AdamW with slightly different weight decay (weight_decay = 0.01 vs 0.001).

    ```
    optimizer = AdamW
    weight_decay = 0.01

    ```

5. lora_alpha, lora_dropout, and LoRA-related params
(Only if you're using LoRA fine-tuning — which Unsloth encourages.)

    🔹 Insight:

    If your language training data is small, you want more dropout to prevent overfitting.
    
    If it's massive, you can reduce dropout for maximum learning speed.
    ```
    lora_alpha = 16
    lora_dropout = 0.1  # Small dataset
    ```
    ```
    lora_alpha = 32
    lora_dropout = 0.05  # Big dataset
    ```

6. gradient_checkpointing

    🔹 Insight:

    Always enable for long-sequence pretraining or fine-tuning, especially for heavy languages.
    ```
    gradient_checkpointing = True
    ```

7. flash_attention

    🔹 Insight:

    Unsloth lets you enable Flash Attention 2 or 3 easily.
    
    Long contexts in fine-tuning multi-sentence Arabic, Hindi documents = critical to save memory and speed.

    ```
    use_flash_attention = True

    ```

🎯 Example full config for Fine-Tuning on Spanish (token-light, large corpus):
train_args = {
    "max_seq_length": 8192,
    "batch_size": 16,
    "gradient_accumulation_steps": 1,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "use_flash_attention": True,
    "gradient_checkpointing": True,
}

🎯 Example full config for Fine-Tuning on Japanese (token-dense, small corpus):
```
train_args = {
    "max_seq_length": 2048,
    "batch_size": 8,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-5,
    "weight_decay": 0.001,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "use_flash_attention": True,
    "gradient_checkpointing": True,
}
```

## PEFT
PEFT (Parameter-Efficient Fine-Tuning) is used during fine-tuning to make large language model (LLM) training more efficient, especially when resources or data are limited.

✅ Why PEFT is Required or Preferred
1. Reduces Memory & Compute Requirements

    Fine-tuning the full model (e.g., 7B+ parameters) requires enormous GPU memory and compute power.
    PEFT (e.g. LoRA) only trains a small subset of additional parameters (like low-rank adapters), often <1% of the full model size.
2. Faster Training

    Because fewer parameters are updated, training is significantly faster.
    You can train on consumer-grade GPUs or more cheaply on cloud GPUs.
3. Modular and Reusable

    You don’t change the base model weights.
    You can swap different PEFT adapters for different tasks without retraining the whole model.
    This enables multi-task or multi-domain systems easily.
4. Smaller Checkpoints

    PEFT checkpoints are often just a few hundred MBs.
    Easy to share and push to Hugging Face Hub.

🛠 When Is PEFT “Required”?

|Scenario|	Is PEFT Required?|	Reason|
| ------ | ----------------- | ------ |
| Full LLM fine-tuning (e.g., 7B)|	✅ Preferred / Often required|	RAM & cost constraints|
| ------ | ----------------- | ------ |
|Tiny model (e.g., DistilGPT2)|	❌ Optional|	Fine-tunning is affordable|
| ------ | ----------------- | ------ |
|Rapid prototyping |	✅ Recommended |	Saves time, cost, and risk|
| ------ | ----------------- | ------ |
|Continual training (e.g., on Wikipedia)|	✅ Yes | For efficiency	|
| ------ | ----------------- | ------ |
|Final production model|	✅ or ❌	|You can merge LoRA into base model or do full fine-tune|
| ------ | ----------------- | ------ |


✅ Unsloth uses huggingface/PEFT under the hood but with lots of speed/memory upgrades.

When you define a PEFT config, you mainly control:

- How much of the model you're adapting
- How fast it adapts
- How safe you are from overfitting

Here's what a PEFTConfig looks like (LoRA example):
```
from peft import LoraConfig

peft_config = LoraConfig(
    r = 16,
    lora_alpha = 32,
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM",
    target_modules = ["q_proj", "v_proj"],  # typical for LLaMA, Mistral, etc
)
```

### Fine tuning PEFT for different languages


1. r (Rank of LoRA Adapters)

    🔹 Insight:

    Small r → fewer trainable parameters → less overfitting (good for small or rare language datasets)

    Big r → more expressiveness → better for rich, large corpora.

    🔹 Tuning:

    |Language Context|	Best r|
    |----------------|--------|
    |Small low-resource lang|	r = 8 or r = 16|
    |-----------------------|------------------|
    |Large corpora (English, Spanish)|	r = 32|
    |------------------|----------------|
        ✅ Example:

        ```
        r = 8   # Amharic fine-tuning
        r = 32  # Spanish or English fine-tuning

        ```

2. lora_alpha

    🔹 Insight:

    Alpha controls the scale of the LoRA weights.
    Higher alpha → smoother updates, avoids sudden weight shifts.
    Important for morphologically complex languages (e.g., Arabic, Finnish).

    🔹 Tuning:

    |Language Context|	Best lora_alpha |
    | -------------- | ---------------  |
    |Morphologically simple (English, French)|	16–32|
    | -------------- | ---------------  |
    |Complex morphology (Arabic, Finnish) |	32–64|
    | -------------- | ---------------  |

    ✅ Example:
    ```
    lora_alpha = 32

    ```

3. lora_dropout

    🔹 Insight:

    - If you're working with small datasets → need more dropout to generalize.
    - If you have huge datasets → you can afford lower dropout.

    🔹 Tuning:

    | Dataset Size|	Best lora_dropout |
    |------------ | ----------------  |
    |Small corpus (e.g., Swahili Wikipedia) |	0.1|
    |------------ | ----------------  |
    |Huge corpus (e.g., English C4)|	0.05 or even 0.01|
    |------------ | ----------------  |
    ✅ Example:
    ```
    lora_dropout = 0.1
    ```


4. target_modules
    
    🔹 Insight:

    - For most transformers like LLaMA, Mistral, GPT models → you LoRA only q_proj and v_proj.
    - If you're fine-tuning deeper models (like some Mistral variations), you can also add k_proj, o_proj.
    
    🔹 Tuning:
    ```
    target_modules = ["q_proj", "v_proj"] # For classic LLaMA/Mistral

    ```


    Or advanced:
    ```
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    ```
    (if you want to adapt attention more heavily — usually needed if your target language has very different syntax)

5. bias

🔹 Insight:

    - Always set bias = "none" unless you're doing full fine-tuning. It just makes LoRA lighter and cleaner.

📦 Full PEFT config examples:

✅ For Hindi-English code-switching fine-tuning (medium corpus):
```
peft_config = LoraConfig(
    r = 16,
    lora_alpha = 32,
    lora_dropout = 0.1,
    bias = "none",
    task_type = "CAUSAL_LM",
    target_modules = ["q_proj", "v_proj"],
)
```
✅ For large Arabic corpus fine-tuning (morphology rich):
```
peft_config = LoraConfig(
    r = 32,
    lora_alpha = 64,
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM",
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"],
)
```
✅ For small Swahili corpus fine-tuning:
peft_config = LoraConfig(
    r = 8,
    lora_alpha = 16,
    lora_dropout = 0.1,
    bias = "none",
    task_type = "CAUSAL_LM",
    target_modules = ["q_proj", "v_proj"],
)

Super Important 💡

→ Unsloth automatically optimizes how LoRA weights are handled (faster fused ops) — so even if your r is big, you won't crash memory like normal Huggingface PEFT.


→ Fine-tuning multilingual models? You can even merge multiple LoRA adapters using Unsloth if you want to fine-tune across multiple languages at once.

 

<u>Below table list the parameters for Unsloth and Peft for Hindi Language fine tunning</u>

| UnSloth Trainer  Parameters| Value | Rational | Observation |
|--------------------------- |-------| -------- | ----------- |
| max_seq_length| 2048 | token dense (one token per character) |  |
|--------------------------- |-------| -------- | ----------- |
| batch_size| 8 | Small batch size for token dense language | |
|--------------------------- |-------| -------- | ----------- |
| learning_rate |  2e-5| Low resource language |  |
|--------------------------- |-------| -------- | ----------- |
| optimizer| AdamW | Morphology complex |  |
|--------------------------- |-------| -------- | ----------- |
| weight_decay| 0.01 | Complex morphology |  |
|--------------------------- |-------| -------- | ----------- |
| lora_alpha| 16 | Small dataset |  |
|--------------------------- |-------| -------- | ----------- |
| lora_dropout| 0.1| Small dataset |  |
|--------------------------- |-------| -------- | ----------- |
| gradient_checkpointing| True | Heavy/Complex morphology |  |
|--------------------------- |-------| -------- | ----------- |
| flash_attention| True | Long context/Heavy/Complex morphology saves memory and improves speed |  |
|--------------------------- |-------| -------- | ----------- |
| num_train_epochs| 3 | Avoid risk of overfitting. Prefer smaller epochs or use early stopping |  |
|--------------------------- |-------| -------- | ----------- |
| max_steps| -1 | Use epoch or set to 1K to 5K |  |
|--------------------------- |-------| -------- | ----------- |
| gradient_accumulation| 4 | Smaller batch size due to GPU limitation |  |
|--------------------------- |-------| -------- | ----------- |
| per_device_train_batch_size| 1 | Memory limitation |  |
|--------------------------- |-------| -------- | ----------- |
| evaluation_strategy| steps | Every 100 to 500 steps |  |
|--------------------------- |-------| -------- | ----------- |
| save_strategy| epoch | Steps or Epoch |  |
|--------------------------- |-------| -------- | ----------- |
| early_stopping_patience| 3 | 2 or 3, stop if loss doesn't decrease |  |
|--------------------------- |-------| -------- | ----------- |



| Peft   Parameters| Value | Rational | Observation |
|----------------- |-------| -------- | ----------- |
| r| 8 | Small corpus |  |
|----------------- |-------| -------- | ----------- |
| lora_alpha| 64 | High alpha smooth updates to weights |  |
|----------------- |-------| -------- | ----------- |
| lora_dropout| 0.1 | Small dataset |  |
|----------------- |-------| -------- | ----------- |
| target_modules| ["q_proj", "k_proj", "v_proj", "o_proj"] | Deeper models Mistral variations |  |
|----------------- |-------| -------- | ----------- |
| bias| None | LoRA lighter and cleaner |  |
|----------------- |-------| -------- | ----------- |
