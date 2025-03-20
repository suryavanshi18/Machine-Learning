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

![image](https://github.com/user-attachments/assets/6500a9e4-43f3-41eb-8c88-8a6b1da829cf)
![image](https://github.com/user-attachments/assets/f9132295-e08d-4f37-b9d9-72834535c6c4)
We freeze the original pre-trained weights (W₀ ∈ ℝ^{d×k}) and create an adaptive low-rank decomposition. 
The weight update ΔW is factorized into two trainable matrices:
- B ∈ ℝ^{d×r} (low-rank projection down)
- A ∈ ℝ^{r×k} (low-rank projection up)

Where the rank r satisfies r ≪ min(d,k). During forward propagation, 
the modified weights become:

W = W₀ + BA

Backpropagation only updates matrices B and A while keeping W₀ frozen.


## Reference
https://www.youtube.com/watch?v=PXWYUTMt-AU&t=812s



