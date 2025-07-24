# LinearAttention Module: Best Practices and Usage Guide

## Introduction

This documentation provides guidelines for integrating the `LinearAttention` module into Pytorch deep learning projects, with emphasis on correct data handling, normalization requirements (crucial for modules using RMSNorm and similar), and architectural considerations for modern efficient attention mechanisms. These recommendations are applicable for time series, NLP, vision, and other sequence modeling tasks.

## Table of Contents

1. [Overview](#overview)
2. [Key Normalization Guidelines](#key-normalization-guidelines)
3. [Model Integration](#model-integration)
4. [Input Formatting](#input-formatting)
5. [Training Workflow](#training-workflow)
6. [Evaluation and Interpretation](#evaluation-and-interpretation)
7. [Extensions and Common Variants](#extensions-and-common-variants)
8. [References](#references)

## Overview

The `LinearAttention` module is a TPTT-inspired, efficient multi-head attention block featuring RMS-based normalization, optional gating via causal average pooling, bidirectional support, and fast linear operator implementations. It is suitable as a drop-in replacement for standard attention in sequence models, provided its input and architectural requirements are met.

## Key Normalization Guidelines

- **RMSNorm Requirement:**
Inputs processed by the LinearAttention block should be normalized (ideally zero mean, unit variance per feature) prior to entering the model, either during data preprocessing or via an input projection + normalization stack.
- **Why:**
RMSNorm divides activations by their root-mean-square without centering; poorly scaled inputs can destabilize gates, SiLU activations and linear attention dynamics.
- **Common tools:**
    - For tabular or time series: `sklearn.preprocessing.StandardScaler`
    - For images: Per-channel normalization to dataset stats
    - For NLP: Embedding standardization, or relying on a pretrained tokenizer and embedding normalization


## Model Integration

### Minimal Example

```python
import torch
import torch.nn as nn
from tptt import LinearAttention

class MySequenceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, num_heads=4, output_dim=1, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attn_layers = nn.ModuleList([
            LinearAttention(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    def forward(self, x, attn_mask=None):
        x = self.input_proj(x)
        for layer in self.attn_layers:
            x = layer(x, attn_mask=attn_mask)
        x = self.norm(x)
        return self.output_proj(x)
```


## Input Formatting

- **Dimensions:**
Input should be `[batch_size, seq_len, input_dim]`.
- **Normalization:**
Normalize input(s) with `StandardScaler` (tabular), per-channel mean/std (vision), or verified embedding normalization (text).
- **Sliding Windows (if sequence-to-point or forecasting):**
Shape input windows accordingly. For standard forecasting, see:

```python
# For time series:
from sklearn.datasets import make_regression
X_raw, y = make_regression(n_samples=n_samples, n_features=1, noise=5)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

def create_dataset(sequence, window_size):
    X_seq, y_seq = [], []
    for i in range(len(sequence) - window_size):
        X_seq.append(sequence[i:i+window_size])
        y_seq.append(sequence[i+window_size])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_dataset(X_scaled, window_size)
```

```python
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # [N, window, 1]
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # [N, 1]
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_seq, y_seq)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

```

- **Projection Layer:**
Always project input to `hidden_dim` before attention layers if input_dim ≠ hidden_dim.


## Training Workflow

- **Optimizer:** Adam/AdamW is recommended.
- **Loss:** MSE for regression/prediction, cross-entropy for classification.
- **Dropout \& Regularization:** Use dropout judiciously inside attention (if specified), especially on small datasets.
- **Batch Size:** Tune depending on GPU memory and sequence length.

Example:

```python
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
model.train()
for epoch in range(50):
    losses = []
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        pred = out[:, -1, :]
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"Epoch {epoch+1}, Loss: {np.mean(losses):.4f}")
```


## Evaluation and Interpretation

- **Prediction Alignment:**
For autoregressive/forecasting, align indices so prediction target and input window match.
- **De-normalization:**
Always inverse-transform model predictions before comparing with targets in the original scale:

```python
y_pred_denorm = scaler.inverse_transform(y_pred.reshape(-1, 1))
```

- **Visualization:**
Compare model outputs with true values, both in normalized and de-normalized forms for interpretability.

```python
import matplotlib.pyplot as plt
plt.plot(time_idx, true_signal, label="True")
plt.plot(time_idx, pred_signal, label="Predicted")
plt.legend(); plt.show()
```

**Note:**
These recommendations ensure model stability, reproducibility, and ease of integration in most modern deep learning pipelines. Always validate normalization and data flow with your specific dataset and task.