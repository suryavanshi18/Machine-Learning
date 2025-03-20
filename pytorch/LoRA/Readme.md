# LoRA (Low-Rank Adaptation of Large Language Models)
LoRA reduces the number of trainable parameters by learning pairs of rank-decompostion matrices while freezing the original weights.
This vastly reduces the storage requirement for large language models adapted to specific tasks and enables 
efficient task-switching during deployment all without introducing inference latency. 


[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Overview
LoRA is a parameter-efficient fine-tuning method that adapts pre-trained models to new tasks **without modifying original weights**. By using low-rank decomposition of weight matrices, it achieves comparable performance to full fine-tuning with **~100-1000x fewer parameters**.

## 🔑 Key Features
- ⚡ **95% fewer parameters** than full fine-tuning
- 🧊 **Frozen base model weights** prevent catastrophic forgetting
- 💾 **10x smaller checkpoint sizes** (stores only adapter weights)
- 🔄 **Plug-and-play adapters** for multi-task serving
- 🎯 **Focuses on attention matrices** (where models need most adaptation)

## 📚 Theoretical Background
For original weight matrix \( W_0 \in \mathbb{R}^{d \times k} \), LoRA represents weight updates as:
\[ \Delta W = W_0 + BA \]
where:
- \( B \in \mathbb{R}^{d \times r} \)
- \( A \in \mathbb{R}^{r \times k} \)
- \( r \ll \min(d,k) \) (low-rank dimension)

![LoRA Architecture](https://miro.medium.com/v2/resize:fit:720/format:webp/1*4S2c6WZrie9E1AD7w2T2SA.png)
