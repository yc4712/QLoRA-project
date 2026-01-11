# LLaMA-2 QLoRA vs LoRA Fine-Tuning + MMLU Evaluation

This project fine-tunes LLaMA-2 7B on the OpenAssistant dataset (OASST1) using two approaches:

1. 4-bit QLoRA

2. Full-precision bf16 LoRA

It compares the two methods on:

- Training efficiency

- GPU memory usage

- Training loss / perplexity

- Downstream performance on MMLU (5-shot)

All experiments were performed in Google Colab using a single GPU.

# Project Structure

```
QLoRA-project/
│
├── notebooks/                # Exploratory development
│   ├── 01_train_lora_vs_qlora.ipynb
│   └── 02_mmlu_eval_lora_vs_qlora.ipynb
│
├── src/                      # Reusable runnable scripts
│   ├── data_prep.py
│   ├── train_qlora_4bit.py
│   ├── train_lora_16bit.py
│   ├── eval_mmlu.py
│   └── training_utils.py
│
├── results/                  # Lightweight experiment artifacts (no checkpoints)
│   ├── 4bit/
│   │   ├── trainer_state.json
│   │   ├── training_logs.json
│   │   ├── training_summary.json
│   │   ├── evaluation_results.json
│   │   └── mmlu_progress_llama2_7b_4bit_qlora.json
│   ├── 16bit/
│   │   ├── trainer_state.json
│   │   ├── training_logs.json
│   │   ├── training_summary.json
│   │   ├── evaluation_results.json
│   │   └── mmlu_progress_llama2_7b_16bit_lora.json
│
├── requirements.txt
└── README.md
```

# Installation

```bash
pip install -r requirements.txt
```

Or install core libs manually:

```bash
pip install transformers bitsandbytes peft accelerate datasets
```

Latest version of bitsandbytes is required for QLoRA to work. GPU required for training and evaluation.

# Data Preparation (OASST1)

Training uses the OpenAssistant Conversations dataset ([OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1)).

Preprocessing steps include:

- Extracting prompts + responses

- Building best conversational turn pairs

- Masking the prompt tokens (-100) so loss is computed only on the assistant reply

- Tokenizing with LLaMA-2 tokenizer

Logic lives in:

```
src/data_prep.py
```

# Training

## 4-bit QLoRA

Quantizes model weights to 4-bit NF4 via bitsandbytes, dramatically reducing memory usage.

Run:

```
python src/train_qlora_4bit.py
```

## 16-bit LoRA

Keeps model weights in bfloat16, applies low-rank adapters on top.

Run:

```
python src/train_lora_16bit.py
```

Both scripts:

- Use Hugging Face Trainer

- Log per-step loss

- Save summaries to results/

# Evaluation - MMLU (5 shot)

MMLU evaluation compares general reasoning ability.

The notebook and script:

```
notebooks/02_mmlu_eval_lora_vs_qlora.ipynb
src/eval_mmlu.py
```

Use:

- LLaMA-2 base model

- Fine-tuned adapters (QLoRA or LoRA)

- 5 example QA demonstrations prepended per prompt

- Log-prob scoring over {A, B, C, D}

# Results Summary

## Training Efficiency

| Model | Peak GPU Memory | Runtime | Final Loss | Final Perplexity |
|:--------:|:---------:|:--------:|:--------:| :--------:|
| **QLoRA (NF4)** | 11.09GB | ~3.3 hrs | ~0.88  | ~2.42  |
| **LoRA (bf16)** | 18.89GB | ~2.7 hrs | ~0.87  | ~2.40 |

## 5-shot MMLU Accuracy

| Model | Accuracy |
|:--------:|:---------:|
| **QLoRA (NF4)**  | ~0.44  |
| **LoRA (bf16)**  | ~0.45  |

Per-step MMLU logs also available in:

```
mmlu_progress_llama2_7b_4bit_qlora.json
mmlu_progress_llama2_7b_16bit_lora.json
```

# Takeaways

- QLoRA achieves ~2× memory savings with comparable accuracy

- LoRA slightly outperforms on MMLU but requires much more GPU

- Both methods successfully fine-tune LLaMA-2 on OASST1 and generalize to reasoning tasks

# Future Works

- Try different fine-tuning datasets such as Alpaca

- Try long-context + instruction-style finetuning

- Evaluate on TruthfulQA, AlpacaEval, GSM8K

# Acknowledgements

## Libraries & Frameworks

- **Hugging Face Transformers & Datasets**

  for model loading, tokenization, and dataset access
  ([HF](https://github.com/huggingface/transformers))

- **PEFT (Parameter-Efficient Fine-Tuning)**

  Edward J. Hu et al., LoRA: Low-Rank Adaptation of Large Language Models, 2021

  Tim Dettmers et al., QLoRA: Efficient Finetuning of Quantized LLMs, 2023

- **BitsAndBytes**

  4-bit quantization backend used with QLoRA ([bitsandbytes](https://github.com/TimDettmers/bitsandbytes))

## Dataset

OpenAssistant Conversations (OASST1)
  accessed via Hugging Face Datasets ([OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1))

## Evaluation Benchmark

MMLU — Massive Multitask Language Understanding: Dan Hendrycks et al., 2021

MMLU JSON files used in evaluation are taken directly from
the official QLoRA GitHub repository:
https://github.com/artidoro/qlora
