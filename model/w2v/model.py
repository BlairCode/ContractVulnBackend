import os
import sys
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
# Added: Import argparse for command-line argument parsing from train.py
import argparse

from . import w2v_save_path, epoch2cp
from .build_vocab import tokenize, build_vocab_from_file, to_ids

# Get the project root directory and related paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model", "w2v")
# Added: Ensure MODEL_DIR exists, inherited from B's improvement
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR, exist_ok=True)

# Word2Vec Skip-Gram model
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_size=50, base=0):
        super(SkipGram, self).__init__()
        if embed_size <= 0:
            raise ValueError("Invalid embed_size, must be positive")
        
        # Embedding layers for target and context words
        self.in_embed = nn.Embedding(vocab_size, embed_size)
        self.out_embed = nn.Embedding(vocab_size, embed_size)
        self.base = base

        # Load pre-trained weights if base > 0
        if base > 0:
            base_path = os.path.join(w2v_save_path, epoch2cp(base))
            if not os.path.exists(base_path):
                raise FileNotFoundError(f"Specified base checkpoint {base_path} not found")
            self.load_state_dict(torch.load(base_path))

    def forward(self, target, pos_context, neg_context):
        # Get embeddings for target, positive context, and negative context
        v = self.in_embed(target)  # (batch_size, embed_size)
        pos_u = self.out_embed(pos_context)  # (batch_size, window_size, embed_size)
        neg_u = self.out_embed(neg_context)  # (batch_size, neg_size, embed_size)
        # Compute scores for positive and negative samples
        pos_score = torch.sum(torch.bmm(pos_u, v.unsqueeze(2)), dim=2)  # (batch_size, window_size)
        neg_score = torch.sum(torch.bmm(neg_u, v.unsqueeze(2)), dim=2)  # (batch_size, neg_size)
        return pos_score, neg_score

    def loss(self, pos_score, neg_score, mask):
        # Compute loss for positive and negative samples
        pos_loss = -F.logsigmoid(pos_score) * mask  # (batch_size, window_size)
        pos_loss = torch.sum(pos_loss, dim=1) / (torch.sum(mask, dim=1) + 1e-6)  # (batch_size,)
        neg_loss = -F.logsigmoid(-neg_score)  # (batch_size, neg_size)
        return pos_loss.mean() + neg_loss.mean()

    def train_sg(self, epoch, target, pos_context, neg_context, mask, lr=1e-4):
        if target.size(0) == 0:
            print("Error: Training data is empty. Cannot train model.")
            return
        optimizer = torch.optim.Adam(self.parameters(), lr)
        print(f"Training started at {datetime.datetime.now()}")  # Log start time
        start_time = time.time()

        for e in range(1, epoch + 1):
            self.train()
            optimizer.zero_grad()
            pos_score, neg_score = self.forward(target, pos_context, neg_context)
            # Fixed: Corrected loss calculation (was neg_context instead of neg_score)
            loss = self.loss(pos_score, neg_score, mask)
            loss.backward()
            optimizer.step()

            if (self.base + e) % 100 == 0:
                epoch_time = time.time() - start_time
                print(f"Epoch {self.base + e}, loss: {loss.item()}")
                print(f"Time taken for epoch: {epoch_time:.2f} seconds")
                start_time = time.time()
                model_path = os.path.join(w2v_save_path, epoch2cp(self.base + e))
                torch.save(self.state_dict(), model_path)

def get_embd(input_data, checkpoint=0):
    # Generate embeddings for input data using the SkipGram model
    vocab = build_vocab_from_file()  # Build vocabulary from file
    sg = SkipGram(len(vocab), base=checkpoint)
    if isinstance(input_data, str):  # Input is a string (code)
        idx = torch.tensor(to_ids(input_data))
    else:  # Input is a list of tokens
        word2idx = {word: idx + 1 for idx, word in enumerate(vocab)}
        word2idx["<pad>"] = 0
        idx = torch.tensor([word2idx.get(token, 0) for token in input_data])
    return sg.in_embed(idx)

def get_embd_from_sg(sg: SkipGram, code):
    # Generate embeddings for code using a pre-trained SkipGram model
    idx = torch.tensor(to_ids(code))
    return sg.in_embed(idx)

# Added: Command-line argument parsing function from train.py
def parse_args():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Script to train SkipGram model")
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train')
    parser.add_argument('--base', type=int, default=0, help='Base checkpoint to start training from')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
    return parser.parse_args()

if __name__ == "__main__":
    # Added: Parse command-line arguments for flexible training, from train.py
    # e.g. python -m model.w2v.model --epochs 500 --base 0 --lr 0.001
    args = parse_args()
    num_epochs = args.epochs
    base = args.base
    learning_rate = args.lr

    vocab = build_vocab_from_file()
    sentences = []
    OUT_DIR = os.path.join(PROJECT_ROOT, "out")
    UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploads")
    found_data = False

    # Load data from out/ directory
    if os.path.exists(OUT_DIR):
        for root, _, files in os.walk(OUT_DIR):
            for file in files:
                if file in ["sliced.txt", "antlr.txt"]:
                    with open(os.path.join(root, file), "r", encoding='utf-8') as f:
                        sentences.extend(line.strip() for line in f if line.strip())
                        found_data = True

    # If no data in out/, load from uploads/ directory
    if not found_data and os.path.exists(UPLOADS_DIR):
        for file in os.listdir(UPLOADS_DIR):
            if file.endswith('.sol'):
                with open(os.path.join(UPLOADS_DIR, file), "r", encoding='utf-8') as f:
                    sentences.extend(line.strip() for line in f if line.strip())
                    found_data = True
                    print(f"Loaded file from uploads: {file}")

    # If still no data, use default sentences
    if not found_data:
        print("No valid data source found (out/ or uploads/). Using default data.")
        sentences = ["function test() { return 1; }", "contract Test { uint x; }"]

    print("sentences:", sentences)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Improve vocabulary handling by adding <unk> token
    word2idx = {word: idx + 1 for idx, word in enumerate(vocab)}
    word2idx["<pad>"] = 0
    word2idx["<unk>"] = len(word2idx)  # Add unknown token
    vocab.add("<pad>")
    vocab.add("<unk>")
    idx2word = {idx: word for word, idx in word2idx.items()}
    print("vocab size:", len(vocab))

    # Prepare training data
    batch_size = sum(len(s.split()) for s in sentences)
    window_size = 2
    neg_size = 4
    targets, pos_context, neg_context, mask = [], [], [], []

    for s in sentences:
        tokens = tokenize(s)
        for i, word in enumerate(tokens):
            target = word2idx.get(word, word2idx["<unk>"])  # Handle unknown tokens
            targets.append(target)
            pos = [word2idx.get(tokens[j], word2idx["<unk>"])
                   for j in range(max(i - window_size, 0), min(i + window_size + 1, len(tokens)))
                   if j != i]
            pad_len = 2 * window_size - len(pos)
            if pad_len > 0:
                pos += [word2idx["<pad>"]] * pad_len
                mask.append([1] * (2 * window_size - pad_len) + [0] * pad_len)
            else:
                mask.append([1] * 2 * window_size)
            pos_context.append(pos)
            neg_context.append(torch.randint(0, len(vocab), (neg_size,)).tolist())

    # Convert data to tensors and move to device
    targets = torch.tensor(targets, dtype=torch.long).to(device)
    pos_context = torch.tensor(pos_context, dtype=torch.long).to(device)
    neg_context = torch.tensor(neg_context, dtype=torch.long).to(device)
    mask = torch.tensor(mask, dtype=torch.float).to(device)
    
    print(f"targets shape: {targets.shape}")
    print(f"pos_context shape: {pos_context.shape}")
    print(f"neg_context shape: {neg_context.shape}")
    print(f"mask shape: {mask.shape}")
    
    # Modified: Use parsed arguments for training, integrating train.py functionality
    sg = SkipGram(len(vocab), base=base).to(device)
    sg.train_sg(num_epochs, targets, pos_context, neg_context, mask, lr=learning_rate)