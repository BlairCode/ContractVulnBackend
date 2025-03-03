import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader, random_split
import os
import time
import datetime
from pathlib import Path

from .w2v import build_vocab
from .w2v import model as w2v_model
from . import epoch2cp, model_save_path, blstm_default_args, cp2epoch
from . import util
from . import dataloader

# Added: Imports from train.py for argument parsing and inference
import sys
import argparse
from .w2v.build_vocab import gen_vocab  # For warmup and inference
from .w2v.model import SkipGram, get_embd_from_sg, get_embd  # For inference support

# Added: For file processing in inference, from train.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ..cmd import process

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "ree")  # Path to the dataset directory

def default_args(args=None):
    if args is None:
        args = blstm_default_args  # Use default arguments from __init__.py
    table = args.copy()
    # Allow external overrides
    if args is not None:
        for key in args:
            table[key] = args[key]
    # Ensure default values for learning rate and epochs
    table.setdefault('lr', 0.0001)
    table.setdefault('epochs', 5)
    return table

class Fusion_Model_BLSTM_ATT(nn.Module):
    def __init__(
            self,
            data=None,  # Training data, optional for inference
            w2v_cp=None,  # Word2Vec checkpoint
            device=None,  # Device (CPU/GPU)
            base=0,  # Base checkpoint for loading pre-trained weights
            args=None,  # Hyperparameters
            inference_only=False  # Flag for inference-only mode
    ):
        super().__init__()
        self.device = device
        self.base = base

        # Hyperparameters
        args = default_args(args)
        self.lr = args['lr']
        self.epochs = args['epochs']
        self.hidden_size = args['hidden_size']
        self.dropout = args['dropout']

        # Initialize training and test datasets if not in inference-only mode
        if not inference_only:
            if data:  # Dynamic data for training
                labels = [label for label, vec in data]
                sequences = [vec.detach() for label, vec in data]
                if not labels:
                    raise ValueError("Training data is empty")
                labels = torch.tensor(labels, dtype=torch.long).to(device)
                padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0).to(device)
                positive_mask = (labels == 1)
                positive_idxs = torch.where(positive_mask)[0]
                negative_idxs = torch.where(~positive_mask)[0]
                if len(positive_idxs) > len(negative_idxs):
                    undersampled_negative_idxs = torch.randperm(len(negative_idxs), device=device)[:len(positive_idxs)]
                    resampled_idxs = torch.cat([positive_idxs, negative_idxs[undersampled_negative_idxs]])
                else:
                    undersampled_positive_idxs = torch.randperm(len(positive_idxs), device=device)[:len(negative_idxs)]
                    resampled_idxs = torch.cat([positive_idxs[undersampled_positive_idxs], negative_idxs])
                x_resampled = padded_sequences[resampled_idxs]
                y_resampled = labels[resampled_idxs]
                dataset = TensorDataset(x_resampled, y_resampled)
                train_size = int(0.8 * len(dataset))
                test_size = len(dataset) - train_size
                if train_size == 0:
                    raise ValueError("Training set has 0 samples")
                self.train_dataset, self.test_dataset = random_split(dataset, [train_size, test_size])
                self.train_loader = DataLoader(self.train_dataset, batch_size=4, shuffle=True)
                self.test_loader = DataLoader(self.test_dataset, batch_size=4, shuffle=True)
                input_size = x_resampled.shape[2]
            elif w2v_cp is not None:  # Use CodeDataset for training
                self.dataset = dataloader.CodeDataset(w2v_cp, device, mode='train')
                self.loader = DataLoader(self.dataset, batch_size=4, shuffle=True)
                self.eval_dataset = dataloader.CodeDataset(w2v_cp, device, mode='eval')
                self.eval_loader = DataLoader(self.eval_dataset, batch_size=4, shuffle=True)
                example_batch, _ = next(iter(self.loader))
                input_size = example_batch.shape[2]
            else:
                raise ValueError("Training mode requires either data or w2v_cp")
        else:
            input_size = 50  # Default input size for inference mode (matches SkipGram embed_size)

        # Model architecture
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.tanh = nn.Tanh()
        self.attention = nn.Linear(self.hidden_size * 2, 1)  # Attention mechanism
        self.softmax = nn.Softmax(dim=1)
        self.classifier = nn.Linear(self.hidden_size * 2, 2)  # Binary classification

        # Load pre-trained weights if base > 0
        if base > 0:
            base_path = os.path.join(model_save_path, epoch2cp(base))
            if not os.path.exists(base_path):
                raise FileNotFoundError(f"Specified base checkpoint {base_path} not found")
            self.load_state_dict(torch.load(base_path))

        self.to(device)

    def forward(self, x):
        x, (_, _) = self.lstm(x)
        scores = self.attention(x)
        scores = self.tanh(scores)
        scores = scores.squeeze(-1)
        attention_weights = self.softmax(scores)
        weights = attention_weights.unsqueeze(-1)
        ctx = torch.sum(weights * x, dim=1)  # Context vector
        logits = self.classifier(ctx)
        return logits

    def train_model(self, epochs=None, lr=None):
        lr = lr or self.lr
        epochs = epochs or self.epochs
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)  # Gradient clipping
        print(f"Training started at {datetime.datetime.now()}")
        start_time = time.time()

        # Select the appropriate data loader
        loader = self.loader if hasattr(self, 'loader') else self.train_loader
        if not loader:
            raise RuntimeError("Model not initialized with training data. Provide data or w2v_cp.")
        for epoch in range(1, epochs + 1):
            self.train()
            total_loss = 0
            for x_batch, y_batch in loader:
                x_batch = x_batch.float()
                y_batch = y_batch.long()
                optimizer.zero_grad()
                logits = self(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0:
                avg_loss = total_loss / len(loader)
                print(f"Epoch {self.base + epoch}, Loss: {avg_loss:.6f}")
                self.eval_model()
                print(f"Time taken for epoch: {time.time() - start_time:.2f} seconds")
                start_time = time.time()
                model_path = os.path.join(model_save_path, epoch2cp(self.base + epoch))
                torch.save(self.state_dict(), model_path)

    def eval_model(self):
        self.eval()
        criterion = nn.CrossEntropyLoss()
        loader = self.eval_loader if hasattr(self, 'eval_loader') else self.test_loader
        if not loader:
            print("Validation data not initialized. Skipping evaluation.")
            return
        with torch.no_grad():
            total_loss = 0
            for x_batch, y_batch in loader:
                x_batch = x_batch.float()
                y_batch = y_batch.long()
                logits = self(x_batch)
                loss = criterion(logits, y_batch)
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            print(f"Validation Loss: {avg_loss:.6f}")

    def predict(self, x):
        self.eval()
        x = x.to(self.device)
        with torch.no_grad():
            x = x.float()
            logits = self(x)
            probs = torch.softmax(logits, dim=1)
        return probs

def parse_args():
    """Parse command-line arguments for training and inference, adapted from train.py."""
    parser = argparse.ArgumentParser(description="Script to manage Fusion_Model_BLSTM_ATT")
    subcmd = parser.add_subparsers(dest='command', help='Sub-command help')

    # Train command
    train_cmd = subcmd.add_parser('train', help='Train BLSTM model')
    train_cmd.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    train_cmd.add_argument('--base', type=int, default=0, help='Base checkpoint to start training from')
    train_cmd.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_cmd.add_argument('--w2v', type=int, default=0, help='Word2Vec checkpoint to generate embeddings from')

    # List command
    list_cmd = subcmd.add_parser('list', help='List available checkpoints')

    # Infer command
    infer_cmd = subcmd.add_parser('infer', help='Infer using model')
    infer_cmd.add_argument("file", type=str, help='.sol file to infer')
    infer_cmd.add_argument('--base', type=int, default=0, help='Base checkpoint to start inference from')
    infer_cmd.add_argument('--w2v', type=int, default=0, help='Word2Vec checkpoint to generate embeddings from')

    return parser.parse_args()

def gen_data():
    """Generate training data from out/processed/vul and out/processed/non_vul directories."""
    data = []
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "out", "processed")
    VULN_DIR = os.path.join(PROCESSED_DIR, "vul")
    NON_VULN_DIR = os.path.join(PROCESSED_DIR, "non_vul")

    if not os.path.exists(PROCESSED_DIR):
        print(f"Processed directory does not exist: {PROCESSED_DIR}")
        return data

    # Process vulnerable samples (vul)
    if os.path.exists(VULN_DIR):
        for file in Path(VULN_DIR).rglob("*_sliced.txt"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    code = f.read().replace('\n', ' ')
                    tokens = build_vocab.tokenize(code)
                    vecs = w2v_model.get_embd(tokens)
                    data.append((1, vecs))  # Label 1 for vulnerable
                print(f"Loaded {file.name} from vul with label 1")
            except Exception as e:
                print(f"Failed to process {file}: {e}")
    else:
        print(f"Vulnerable directory does not exist: {VULN_DIR}")

    # Process non-vulnerable samples (non_vul)
    if os.path.exists(NON_VULN_DIR):
        for file in Path(NON_VULN_DIR).rglob("*_sliced.txt"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    code = f.read().replace('\n', ' ')
                    tokens = build_vocab.tokenize(code)
                    vecs = w2v_model.get_embd(tokens)
                    data.append((0, vecs))  # Label 0 for non-vulnerable
                print(f"Loaded {file.name} from non_vul with label 0")
            except Exception as e:
                print(f"Failed to process {file}: {e}")
    else:
        print(f"Non-vulnerable directory does not exist: {NON_VULN_DIR}")

    positive_samples = sum(1 for label, _ in data if label == 1)
    negative_samples = len(data) - positive_samples
    print(f"Positive samples (vulnerable): {positive_samples}, Negative samples (non-vulnerable): {negative_samples}")
    if not data:
        print("No training data generated. Check out/processed directory.")
    else:
        print(f"Generated {len(data)} training samples.")
    return data

def warmup(base: int, w2v_cp: int):
    """
    For efficiency, pre-load the model and word2vec model, from train.py
    """
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    device = torch.device('cuda' if cuda_available else ('mps' if mps_available else 'cpu'))
    model = Fusion_Model_BLSTM_ATT(
        w2v_cp=w2v_cp,
        device=device,
        base=base,
        inference_only=True
    )
    vocab, _ = gen_vocab()
    sg = SkipGram(len(vocab), base=w2v_cp)
    return model, sg

def analyze_file(model: Fusion_Model_BLSTM_ATT, w2v_model: SkipGram, fp: str):
    """
    Utilize the pre-loaded model and word2vec model to analyze a file, from train.py
    """
    if not os.path.exists(fp):
        raise FileNotFoundError(f"File {fp} not found")

    process.process(filepath=fp)
    files = os.listdir('eval')

    # Assert that both files are present
    if not 'sliced.txt' in files:
        raise FileNotFoundError("sliced.txt not found")
    if not 'antlr.txt' in files:
        raise FileNotFoundError("antlr.txt not found")

    code = ''
    sliced_path = os.path.join('eval', 'sliced.txt')
    with open(sliced_path, 'r') as f:
        s_code = f.read().replace('\n', ' ')
        code += s_code
    code += ' '
    antlr_path = os.path.join('eval', 'antlr.txt')
    with open(antlr_path, 'r') as f:
        a_code = f.read().replace('\n', ' ')
        code += a_code

    vecs = get_embd_from_sg(w2v_model, code)
    result = model.predict(vecs)
    return result.tolist()[0][1]  # Return probability of vulnerability (class 1)

if __name__ == "__main__":
    # Modified: Parse arguments and handle multiple commands, integrating train.py functionality
    args = parse_args()
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    mps_available = torch.backends.mps.is_available()
    print(f"MPS Available: {mps_available}")
    device = torch.device('cuda' if cuda_available else ('mps' if mps_available else 'cpu'))
    print(f"Using device: {device}")

    if args.command == 'train':
        num_epochs = args.epochs
        base = args.base
        learning_rate = args.lr
        w2v_cp = args.w2v

        data = gen_data()
        if not data:
            print("No valid data generated. Falling back to CodeDataset.")
            model = Fusion_Model_BLSTM_ATT(w2v_cp=w2v_cp, device=device, base=base)
        else:
            model = Fusion_Model_BLSTM_ATT(data=data, device=device, base=base)
        model.to(device)
        model.train_model(epochs=num_epochs, lr=learning_rate)
        torch.save(model.state_dict(), os.path.join(PROJECT_ROOT, "model", "fusion_model.pt"))
        print(f"Model saved to: {os.path.join(PROJECT_ROOT, 'model', 'fusion_model.pt')}")
    
    elif args.command == 'list':
        # Added: List available checkpoints, from train.py
        print("Listing available checkpoints")
        cps = []
        for cp in os.listdir(model_save_path):
            epoch = cp2epoch(cp)
            if epoch is not None:
                cps.append(epoch)
        if len(cps) == 0:
            print("No checkpoints found, try training a model first")
        else:
            print("Available checkpoints:")
            for cp in cps:
                print(cp)
    
    elif args.command == 'infer':
        # Added: Inference logic for .sol files, from train.py
        base = args.base
        w2v_cp = args.w2v
        fp = args.file
        model, sg = warmup(base, w2v_cp)
        result = analyze_file(model, sg, fp)
        print(f"Inference result (vulnerability probability): {result}")
    
    else:
        print("No valid command provided. Use --help for options.")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    mps_available = torch.backends.mps.is_available()
    print(f"MPS Available: {mps_available}")
    device = torch.device('cuda' if cuda_available else ('mps' if mps_available else 'cpu'))
    print(f"Using device: {device}")

    data = gen_data()
    if not data:
        print("No valid data generated. Falling back to CodeDataset.")
        model = Fusion_Model_BLSTM_ATT(w2v_cp=0, device=device)  # w2v_cp=0 for initial checkpoint
    else:
        model = Fusion_Model_BLSTM_ATT(data=data, device=device)
    model.to(device)
    model.train_model()
    torch.save(model.state_dict(), os.path.join(PROJECT_ROOT, "model", "fusion_model.pt"))
    print(f"Model saved to: {os.path.join(PROJECT_ROOT, 'model', 'fusion_model.pt')}")