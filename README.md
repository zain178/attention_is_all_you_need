# Attention Is All You Need: PyTorch Transformer Implementation

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green)]()

## 🚀 Project Overview

This repository contains a from-scratch PyTorch implementation of the original Transformer architecture from Vaswani et al.’s paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). Building it myself was the hardest thing I’ve ever done—I still don’t fully grok every detail—but it deepened my understanding of modern sequence-to-sequence models (GPT, BERT, etc.). Along the way I learned how embeddings work, how the encoder uses Queries, Keys, and Values to compute self-attention, how the decoder runs in parallel with encoder outputs, and how softmaxed attention scores drive next-token prediction.

## 🎯 Key Learnings

1. **Tensor Shape Management**  
   Splitting `d_model` into `h` attention heads and reshaping tensors (`view` + `transpose`) without mismatches is critical—one wrong dimension and you get a nasty “shape invalid” error.

2. **Modular Code Design**  
   Breaking the model into clear components (Embeddings, PositionalEncoding, MultiHeadAttention, FeedForward, Residual + LayerNorm, Encoder/Decoder blocks) makes it far easier to debug, test, and extend.

3. **Iterative Debugging & Validation**  
   Inspecting tensor shapes at each stage, printing intermediate outputs, and writing small unit tests for each submodule saved countless hours chasing silent shape or type bugs.

## 📂 Repository Structure

.
├── README.md ← this file
├── model.py ← Transformer implementation
├── dataset.py ← data loading & masking logic
├── train.py ← training loop, TensorBoard logging
├── config.py ← hyperparameters & file paths
├── requirements.txt ← Python dependencies
└── runs/ ← TensorBoard logs


## ⚙️ Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/transformer-pytorch.git
   cd transformer-pytorch


📐 Architecture
InputEmbeddings + PositionalEncoding

N× Encoder Blocks (Self-Attention → Add&Norm → FeedForward → Add&Norm)

N× Decoder Blocks (Masked Self-Attention → Add&Norm → Cross-Attention → Add&Norm → FeedForward → Add&Norm)

ProjectionLayer with log-softmax over target vocabulary

Refer to the comments in model.py for detailed method-level explanations.
