# empathetic-dialogue-transformer
Empathetic Dialogue Transformer

This repository contains the code for building, training, and deploying a Transformer-based chatbot designed to generate empathetic responses. The model is built from scratch using PyTorch and is trained on the "Empathetic Dialogues" dataset from Facebook AI. The project includes a user-friendly interface created with Streamlit for real-time interaction.

✨ Features

End-to-End Implementation: From data preprocessing to a deployable UI.

Transformer from Scratch: A complete implementation of the original "Attention Is All You Need" architecture in PyTorch.

Interactive Chatbot UI: A web-based interface built with Streamlit to chat with the trained model.

Custom Tokenizer: A Byte-Pair Encoding (BPE) tokenizer trained specifically on the dialogue data.

Multiple Decoding Strategies:

Greedy Search: For fast, deterministic responses.

Beam Search: For potentially more coherent and higher-quality responses.

Attention Visualization: An optional attention heatmap to see which input words the model "focuses on" when generating a response.

🏛️ Model Architecture

The model is a standard encoder-decoder Transformer. It leverages self-attention and cross-attention mechanisms to understand the context of a given situation and emotion, and then generates a relevant, empathetic reply.

Embedding Layer: Converts input tokens into dense vectors.

Positional Encoding: Injects information about the order of tokens.

Encoder Stack: 2 Encoder Layers to process the input sequence.

Decoder Stack: 2 Decoder Layers to generate the output sequence.

Final Linear Layer: A softmax layer to predict the next token from the vocabulary.

📚 Dataset

This model is trained on the Empathetic Dialogues dataset, a large-scale dataset of 25,000 conversations grounded in emotional situations.

Source: Facebook AI

Citation: Rashkin, Hannah, et al. "Towards empathetic open-domain conversation models: A new benchmark and dataset." arXiv preprint arXiv:1811.00207 (2018).

🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

Prerequisites

Python 3.9 or higher

pip (Python package installer)

Installation

Clone the repository:

git clone(https://github.com/Usman3660/empathetic-dialogue-transformer.git)
cd empathetic-dialogue-transformer


Create and activate a virtual environment (recommended):

python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate


Install the required dependencies:

pip install -r requirements.txt


(Note: You will need to create a requirements.txt file containing libraries like torch, streamlit, tokenizers, pandas, numpy, sacrebleu, matplotlib)

⚙️ How to Use

The project runs in two main stages: training the model and then launching the user interface.

Stage 1: Preprocessing & Training

First, you need to run the complete training pipeline. This will process the dataset and generate two crucial files: tokenizer.json and best_model.pt.

Download the Empathetic Dialogues dataset and extract it.

Place the emotion-emotion_69k.csv file in your project directory.

Run the training script (e.g., train.py). This script should contain the data preprocessing, model definition, and training loop.

python train.py


This process can take a long time, depending on your hardware. Upon completion, you will have the model weights (best_model.pt) and the tokenizer (tokenizer.json).

Stage 2: Launch the Chatbot UI

Once the model and tokenizer files are generated, you can launch the interactive Streamlit application.

Make sure app.py, best_model.pt, and tokenizer.json are all in the root directory.

Run the following command in your terminal:

streamlit run app.py


Your default web browser will open a new tab with the chatbot interface ready to use!

📂 Project Structure

empathetic-dialogue-transformer/
├── .gitignore
├── README.md
├── requirements.txt
├── train.py               # Main script for preprocessing and training
├── app.py                 # The Streamlit chatbot application
├── best_model.pt          # (Generated after training) Trained model weights
└── tokenizer.json         # (Generated after training) BPE tokenizer file


📄 License

This project is licensed under the MIT License. See the LICENSE file for details.
