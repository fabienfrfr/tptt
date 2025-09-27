---
marp: true
theme: default
class: lead
paginate: true
---

# TPTT: Transforming Pretrained Transformers into Titans

Fabien Furfaro
September 2025

---

## Context and Motivation

- Transformer LLMs reach strong NLP performance  
- Quadratic complexity of self-attention limits long-context applications  
- Need efficient retrofit solutions preserving pretrained weights

---

## Objectives

- Retrofit pretrained Transformers with:  
  - Linearized attention (LiZA)  
  - Memory as Gate (MaG) gating  
- Support parameter-efficient fine-tuning (LoRA)  
- Maintain compatibility with Hugging Face Transformers

---

## Linearized Attention

Standard attention complexity:

$$
O(T^2 D), \quad \text{where } T=\text{sequence length}, D=\text{hidden dim}
$$

Linear attention output at position $ t $:

$$
O_{lin, t} = \frac{\sum_{i=1}^t \phi(q_t) \phi(k_i)^T v_i}{\sum_{i=1}^t \phi(q_t) \phi(k_i)^T}
$$

- $\phi$: feature map (e.g. ELU or SiLU activation)  
- $q_t, k_i, v_i$: query, key, value vectors  

Normalization and gating applied for training stability.

---

## Memory as Gate (MaG)

Gated mixing of softmax and linear attention outputs:

$$
O_{MaG} = \alpha \cdot O_{base} + (1-\alpha) \cdot O_{lin}
$$

or cross-gate mixing:

$$
O_{MaG} = \alpha \cdot O_{base} + (1-\alpha) \cdot O_{lin} + \alpha (1-\alpha) O_{base} \odot O_{lin}
$$

- $\alpha \in [0,1]$ is a gating parameter  
- $\odot$ denotes elementwise multiplication  
- Enables adaptive combination balancing efficiency and expressivity

---

## DeltaProduct Operator

Extended parallel state update with multiple Householder transformations:

$$
H_{i,j} = H_{i,j-1}(I - v_{i,j} k_{i,j}^T) + k_{i,j} v_{i,j}^T, \quad j=1,\ldots,n_h
$$

- $H_{i,0} = H_{i-1,n_h}$ initialization from previous state  
- $k_{i,j}, v_{i,j}$: key and value vectors for $j$-th update  
- $n_h$: number of Householder updates, controls expressivity

Recursive expansion expresses memory as product of these transformations enabling parallel computation.

---

## Virtual Token Expansion

To facilitate $n_h$ Householder updates per token, each real token $x_t \in \mathbb{R}^D$ is expanded into $n_h$ virtual tokens:

- Derivative trick:

$$
x_{t, deriv}^{(m)} = \sum_{k=0}^m (-1)^k \binom{m}{k} x_{t-k}
$$

- Rotary trick applies phase rotations $R_m$ to features:

$$
x_{t, rot}^{(m)} = R_m x_t
$$

This enriches token representations for more expressive memory updates.

---

## Experimental Setup

- Models: Llama, Qwen, OpenELM, Mistral (1B-7B params)  
- Dataset: Alpaca, 5 epochs, max 512 tokens  
- Metrics: MMLU Exact Match, Partial Match  

---

## Key Results Summary

- Up to 20% relative improvement in Exact Match over baselines  
- More Householder updates improve convergence and accuracy  
- Combination of derivative and rotary tricks beneficial  
- LoRA accelerates early fine-tuning phases  
- Memory gating $\alpha \approx 0.5$ effective for stable training

---

## Conclusion and Future Work

- TPTT efficiently retrofits pretrained LLMs with memory augmentation  
- Balances expressivity and efficiency without full retraining  
- Scales well across architectures and model sizes  
- Future: optimize efficiency further, extend to larger models and diverse tasks

---

## References & Resources

- GitHub: github.com/fabienfr/frtptt  
- PyPI: pypi.org/project/tptt  
- Relevant papers: DeltaProduct, Titans, LoRA

---

## Thank you

Questions?
