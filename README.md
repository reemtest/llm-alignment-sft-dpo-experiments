# llm-alignment-sft-dpo-experiments
# LLM Code Generation and Guarding (FFT, SFT, DPO)

## 📌 Assignment Overview
This project explores **post-training alignment of Large Language Models (LLMs)** by implementing and comparing three training approaches:

- Full Fine-Tuning (FFT)
- Supervised Fine-Tuning (SFT) with QLoRA
- Direct Preference Optimization (DPO)

The goal is to analyze the trade-off between:
- Code generation capability
- Behavioral safety and alignment

---

# Part I: Full Fine-Tuning (FFT)

## Dataset / Model
- Base Model: `xlm-roberta-base` or `xlm-roberta-large`
- Task: Language modeling adaptation for code generation

## Implementation Steps
- Load RoBERTa with LM head
- Tokenization and dataset preparation
- Train on full dataset (SFT-style causal LM objective applied via adaptation)
- Gradient checkpointing for memory efficiency

## Training Configuration
- Batch size: 4  
- Epochs: 2  
- Learning rate: 5e-5  
- Optimizer: AdamW  
- bf16: True  

## Output
- Fine-tuned RoBERTa model
- Baseline code generation capability

---

# Part II: Supervised Fine-Tuning (SFT) with QLoRA

## Model
- `Qwen/Qwen2-1.5B-Instruct`

## Dataset
- `flytech/python-codes-25k`

## Implementation Steps
- Apply 4-bit quantization (NF4)
- Configure LoRA adapters
- Fine-tune model on Python code instruction dataset
- Train using causal LM loss on response tokens only

## LoRA Configuration
- Rank (r): 16  
- Target modules: all-linear  
- Max length: 1024  

## Training Configuration
- Batch size: 1  
- Gradient accumulation: 4  
- Epochs: 1  
- Learning rate: 2e-4  
- Optimizer: paged_adamw_8bit  

## Output
- Code-specialized Qwen model (SFT version)

---

# Part III: Direct Preference Optimization (DPO)

## Dataset
- `jondurbin/truthy-dpo-v0.1`

## Objective
Align model behavior using preference pairs:
- Chosen response (preferred)
- Rejected response (non-preferred)

## Implementation Steps
- Load SFT-trained Qwen model as reference
- Apply DPO loss:
  - Compare probability of chosen vs rejected outputs
  - Optimize preference alignment
- Train across multiple β values

## DPO Loss Function
- Controls alignment strength using β parameter

## Training Configuration
- Batch size: 1  
- Gradient accumulation: 4  
- Epochs: 3  
- Learning rate: 1e-5  
- Max length: 512  
- β values: 0.1, 0.5, 0.8, 1.0  

## Output
- Behaviorally aligned LLM
- Controlled refusal and safer responses

---

# Evaluation

## Metrics Compared Across All Models

### 1. Code Generation Quality
- Syntax correctness
- Functional correctness
- Task completion rate

### 2. Alignment Behavior
- Refusal of unsafe requests
- Instruction compliance
- Safety consistency

### 3. Trade-off Analysis
- Capability vs safety balance
- Effect of DPO β parameter

---

# Tools & Frameworks
- Hugging Face Transformers
- PEFT (LoRA / QLoRA)
- TRL (DPO training)
- BitsAndBytes (4-bit quantization)
- Weights & Biases (optional logging)

---

# Key Insight
This project demonstrates how:
- FFT improves general adaptation
- SFT enables strong code generation ability
- DPO enforces safety and behavioral alignment

### 🔗 Collaborators

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/Malak Aboouf">
        <img src="https://github.com/malakaboouf.png" width="100px;" alt="Malak Aboouf"/><br />
        <sub><b>Malak Aboouf</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/reemtest">
        <img src="https://github.com/reemtest.png" width="100px;" alt="reemtest"/><br />
        <sub><b>reemtest</b></sub>
      </a>
    </td>
  </tr>
</table>

