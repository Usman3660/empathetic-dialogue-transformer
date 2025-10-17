import streamlit as st
import torch
import torch.nn as nn
import math
import re
from tokenizers import Tokenizer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- 1. MODEL ARCHITECTURE DEFINITION ---
# This section contains the PyTorch model classes, copied from the training script.
# It's necessary to define the architecture before loading the saved model weights.

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.attention_weights = None

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q = self.q_linear(q).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        self.attention_weights = attn # Store attention weights
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        return self.out_linear(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn = self.mha(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn))
        ff = self.ff(x)
        return self.norm2(x + self.dropout(ff))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.self_mha = MultiHeadAttention(d_model, num_heads)
        self.cross_mha = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        self_attn = self.self_mha(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn))
        cross_attn = self.cross_mha(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(cross_attn))
        ff = self.ff(x)
        return self.norm3(x + self.dropout(ff))

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=2, num_enc_layers=2, num_dec_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dropout) for _ in range(num_enc_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dropout) for _ in range(num_dec_layers)])
        self.linear = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.dropout(self.pos_enc(self.embedding(src)))
        tgt_emb = self.dropout(self.pos_enc(self.embedding(tgt)))
        enc_out = src_emb
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask)
        dec_out = tgt_emb
        for i, layer in enumerate(self.decoder_layers):
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)
        return self.linear(dec_out)

# --- 2. SETUP AND UTILITY FUNCTIONS ---

# Use @st.cache_resource to load model and tokenizer only once
@st.cache_resource
def load_model_and_tokenizer():
    """Loads the trained Transformer model and tokenizer."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = Tokenizer.from_file("tokenizer.json")
        
        # Instantiate model with parameters from training
        vocab_size = tokenizer.get_vocab_size()
        model = Transformer(vocab_size=vocab_size, d_model=256, num_heads=2, num_enc_layers=2, num_dec_layers=2)
        
        # Load the saved state dictionary
        model.load_state_dict(torch.load('best_model.pt', map_location=device))
        model.to(device)
        model.eval()
        return model, tokenizer, device
    except FileNotFoundError:
        st.error("Model or tokenizer file not found. Please ensure 'best_model.pt' and 'tokenizer.json' are in the same directory.")
        return None, None, None

def normalize_text(text):
    """Cleans and standardizes text."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def create_masks(src, tgt, pad_id, device):
    """Creates masks for the Transformer model."""
    src_mask = (src != pad_id).unsqueeze(1).unsqueeze(2).to(device)
    tgt_len = tgt.size(1)
    tgt_mask = (torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1) == 0).to(device)
    tgt_pad_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(2).to(device)
    tgt_mask = tgt_mask & tgt_pad_mask
    return src_mask, tgt_mask

# --- 3. DECODING STRATEGIES ---

def greedy_decode(model, src, max_len=50, bos_id=2, eos_id=3, device='cpu'):
    """Greedy decoding: selects the most likely token at each step."""
    src = src.to(device)
    src_mask = (src != model.embedding.padding_idx).unsqueeze(1).unsqueeze(2).to(device) if model.embedding.padding_idx is not None else None

    with torch.no_grad():
        src_emb = model.dropout(model.pos_enc(model.embedding(src)))
        enc_out = src_emb
        for layer in model.encoder_layers:
            enc_out = layer(enc_out, src_mask)
        
        ys = torch.ones(1, 1).fill_(bos_id).type(torch.long).to(device)
        for _ in range(max_len - 1):
            tgt_mask = (torch.triu(torch.ones(ys.size(1), ys.size(1)), diagonal=1) == 0).to(device)
            out = model(src, ys, src_mask, tgt_mask)
            prob = out[:, -1]
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
            if next_word == eos_id:
                break
    return ys

def beam_search_decode(model, src, max_len=50, beam_width=5, bos_id=2, eos_id=3, device='cpu'):
    """Beam search decoding: keeps track of k most likely sequences."""
    src = src.to(device)
    src_mask = (src != model.embedding.padding_idx).unsqueeze(1).unsqueeze(2).to(device) if model.embedding.padding_idx is not None else None

    with torch.no_grad():
        src_emb = model.dropout(model.pos_enc(model.embedding(src)))
        enc_out = src_emb
        for layer in model.encoder_layers:
            enc_out = layer(enc_out, src_mask)

        # Start with <bos> token
        sequences = [[torch.tensor([bos_id], device=device), 0.0]]

        for _ in range(max_len):
            all_candidates = []
            for seq, score in sequences:
                if seq[-1].item() == eos_id:
                    all_candidates.append([seq, score])
                    continue
                
                tgt_mask = (torch.triu(torch.ones(seq.size(0), seq.size(0)), diagonal=1) == 0).to(device)
                out = model(src, seq.unsqueeze(0), src_mask, tgt_mask)
                prob = torch.log_softmax(out[:, -1], dim=-1)
                
                topk_scores, topk_words = prob.topk(beam_width, dim=-1)

                for i in range(beam_width):
                    next_tok, next_score = topk_words[0][i], topk_scores[0][i]
                    new_seq = torch.cat([seq, next_tok.unsqueeze(0)])
                    new_score = score + next_score.item()
                    all_candidates.append([new_seq, new_score])
            
            ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
            sequences = ordered[:beam_width]
            
            # Stop if all top sequences end with <eos>
            if all(s[0][-1].item() == eos_id for s in sequences):
                break

    return sequences[0][0].unsqueeze(0)


# --- 4. ATTENTION VISUALIZATION ---

def get_attention_weights(model, src, generated_seq, pad_id, device):
    """Performs a forward pass to capture attention weights."""
    model.eval()
    with torch.no_grad():
        src_mask, tgt_mask = create_masks(src, generated_seq, pad_id, device)
        
        src_emb = model.dropout(model.pos_enc(model.embedding(src)))
        tgt_emb = model.dropout(model.pos_enc(model.embedding(generated_seq)))
        
        enc_out = src_emb
        for layer in model.encoder_layers:
            enc_out = layer(enc_out, src_mask)
            
        dec_out = tgt_emb
        # We want the cross-attention from the LAST decoder layer
        for i, layer in enumerate(model.decoder_layers):
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)
        
        # Access the stored weights from the last decoder's cross-attention module
        attention = model.decoder_layers[-1].cross_mha.attention_weights
        return attention

def plot_attention_heatmap(weights, src_tokens, tgt_tokens):
    """Plots and displays the attention heatmap."""
    fig, ax = plt.subplots(figsize=(10, 10))
    # Squeeze to remove batch and head dimensions, then average over heads
    weights = weights.squeeze(0).cpu().numpy()
    if weights.ndim > 2:
        weights = weights.mean(axis=0) # Average over heads

    cax = ax.matshow(weights, cmap='bone')
    fig.colorbar(cax)

    ax.set_xticklabels([''] + src_tokens, rotation=90)
    ax.set_yticklabels([''] + tgt_tokens)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    st.pyplot(fig)


# --- 5. STREAMLIT UI ---

st.set_page_config(layout="wide")
st.title("🤖 Empathetic Chatbot")
st.markdown("An interface to interact with a Transformer model trained on the Empathetic Dialogues dataset.")

model, tokenizer, device = load_model_and_tokenizer()

if model is not None:
    # Get available emotions from the tokenizer's special tokens
    emotions = [
        tok.replace("<emotion_", "").replace(">", "") 
        for tok in tokenizer.get_vocab().keys() if tok.startswith("<emotion_")
    ]
    emotions = sorted(list(set(emotions))) # Get unique sorted list

    # Initialize session state for conversation history
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Sidebar for options
    with st.sidebar:
        st.header("Inference Options")
        selected_emotion = st.selectbox("Select an Emotion (optional)", ["none"] + emotions)
        decoding_strategy = st.radio("Decoding Strategy", ["Greedy Search", "Beam Search"])
        
        beam_width = 5
        if decoding_strategy == "Beam Search":
            beam_width = st.slider("Beam Width", min_value=2, max_value=10, value=5)
        
        show_attention = st.checkbox("Show Attention Heatmap")

    # Main chat interface
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("How are you feeling today?"):
        # Add user message to history
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare model input
        cleaned_prompt = normalize_text(prompt)
        if selected_emotion != "none":
            input_text = f"Emotion: {selected_emotion} | Situation: {cleaned_prompt} Agent:"
        else:
            input_text = f"Situation: {cleaned_prompt} Agent:"
        
        input_ids = tokenizer.encode(f"<bos> {input_text}").ids
        src = torch.tensor([input_ids], device=device)

        # Generate response based on selected strategy
        with st.spinner("Thinking..."):
            if decoding_strategy == "Greedy Search":
                output_ids = greedy_decode(model, src, bos_id=tokenizer.token_to_id("<bos>"), eos_id=tokenizer.token_to_id("<eos>"), device=device)
            else: # Beam Search
                output_ids = beam_search_decode(model, src, beam_width=beam_width, bos_id=tokenizer.token_to_id("<bos>"), eos_id=tokenizer.token_to_id("<eos>"), device=device)

        # Decode and display response
        response_text = tokenizer.decode(output_ids.squeeze(0).tolist(), skip_special_tokens=True).strip()
        st.session_state.history.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)

            # Display attention heatmap if requested
            if show_attention:
                with st.expander("See Attention Weights"):
                    src_tokens = tokenizer.encode(f"<bos> {input_text}").tokens
                    tgt_tokens = tokenizer.decode(output_ids.squeeze(0).tolist()).split()
                    
                    attention_weights = get_attention_weights(model, src, output_ids, tokenizer.token_to_id("<pad>"), device)
                    plot_attention_heatmap(attention_weights, src_tokens, tgt_tokens)
