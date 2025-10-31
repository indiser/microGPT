# MicroGPT

A fine-tuned GPT-2 model using LoRA (Low-Rank Adaptation) for efficient text generation.

## Overview

This project demonstrates how to fine-tune the GPT-2 model from Hugging Face using LoRA adapters, making it computationally efficient while maintaining performance. The model is trained on custom data and deployed with a Gradio interface for easy interaction.

## Features

- **LoRA Fine-tuning**: Uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA adapters
- **Custom Data Training**: Supports training on JSONL and text files
- **Interactive Interface**: Gradio web app for text generation
- **GPU Acceleration**: Optimized for CUDA when available

## Files

- `microGPT.ipynb` - Main Jupyter notebook containing the complete training pipeline
- `gradio_app.py` - Gradio web interface for the fine-tuned model
- `README.md` - Project documentation

## Requirements

```bash
pip install transformers peft datasets gradio torch
```

## Usage

### Training the Model

1. Prepare your training data:
   - `train.jsonl` - JSON Lines file with prompt-response pairs
   - `microGPT.txt` - Plain text file with additional training data

2. Run the notebook `microGPT.ipynb` to:
   - Load and preprocess data
   - Configure LoRA parameters
   - Fine-tune GPT-2 model
   - Save the trained model

### Running the Interface

```bash
python gradio_app.py
```

This launches a web interface where you can input prompts and generate text using the fine-tuned model.

## Model Configuration

- **Base Model**: GPT-2 (openai-community/gpt2)
- **LoRA Parameters**:
  - Rank (r): 8
  - Alpha: 32
  - Dropout: 0.1
  - Trainable parameters: 294,912 (0.24% of total)

## Training Details

- **Epochs**: 3
- **Batch Size**: 4
- **Learning Rate**: 2e-4
- **Max Sequence Length**: 128 tokens
- **Training Loss**: Decreased from 7.98 to 2.77

## Performance

The LoRA approach significantly reduces computational requirements by training only 0.24% of the model parameters while maintaining text generation quality.