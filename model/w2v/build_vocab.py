import os

# Define sets of operators (unchanged)
operators1 = {
    '(', ')', '[', ']', '.',
    '+', '-', '*', '&', '/',
    '%', '<', '>', '^', '|',
    '=', ',', '?', ':', ';',
    '{', '}'
}
operators2 = {
    '->', '++', '--',
    '!~', '<<', '>>', '<=', '>=',
    '==', '!=', '&&', '||', '+=',
    '-=', '*=', '/=', '%=', '&=', '^=', '|='
}
operators3 = {'<<=', '>>='}

# Project paths (retaining B's design)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "out")
VOCAB_FILE = os.path.join(PROJECT_ROOT, "model", "w2v", "vocab.txt")
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploads")

def tokenize(line):
    """Tokenize a line of code into tokens."""
    tmp, w = [], []
    i = 0
    while i < len(line):
        if line[i] == ' ':
            tmp.append(''.join(w))
            tmp.append(line[i])
            w = []
            i += 1
        elif line[i:i + 3] in operators3:
            tmp.append(''.join(w))
            tmp.append(line[i:i + 3])
            w = []
            i += 3
        elif line[i:i + 2] in operators2:
            tmp.append(''.join(w))
            tmp.append(line[i:i + 2])
            w = []
            i += 2
        elif line[i] in operators1:
            tmp.append(''.join(w))
            tmp.append(line[i])
            w = []
            i += 1
        else:
            w.append(line[i])
            i += 1
    res = list(filter(lambda c: c != '', tmp))
    return list(filter(lambda c: c != ' ', res))

def gen_vocab():
    """Generate vocabulary and word2idx, restoring A's functionality."""
    special_tokens = ['<pad>', '<unk>']  # Restore A's special tokens
    vocab = build_vocab_from_file()  # B's vocabulary base
    vocab.add('\n')
    vocab.add(' ')
    word2idx = {word: idx + len(special_tokens) for idx, word in enumerate(vocab)}
    for i, st in enumerate(special_tokens):
        word2idx[st] = i
        vocab.add(st)
    return vocab, word2idx

def to_ids(input_data):
    """Convert code or tokens to IDs, supporting <unk>, compatible with A and B."""
    vocab, word2idx = gen_vocab()
    if isinstance(input_data, str):  # A's code input
        tokens = tokenize(input_data)
    else:  # B's token input
        tokens = input_data
    return [word2idx.get(token, word2idx['<unk>']) for token in tokens]

def build_vocab_from_code():
    """Build vocabulary from out/ or uploads/ directory, retaining B's dynamism and improving number support."""
    vocab = set()
    vocab.add('\n')
    vocab.add(' ')
    
    found_data = False
    if os.path.exists(OUT_DIR):
        for root, _, files in os.walk(OUT_DIR):
            for file in files:
                if file in ['antlr.txt', 'sliced.txt']:
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read().replace('\n', ' ')
                        tokens = tokenize(code)
                        vocab.update(tokens)
                        found_data = True
                else:
                    print(f"Skipping file: {file}")

    if not found_data and os.path.exists(UPLOADS_DIR):
        for file in os.listdir(UPLOADS_DIR):
            if file.endswith('.sol'):
                file_path = os.path.join(UPLOADS_DIR, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read().replace('\n', ' ')
                    tokens = tokenize(code)
                    vocab.update(tokens)
                    found_data = True
                    print(f"Loaded file from uploads: {file}")

    if not found_data:
        print("No valid data source found (out/ or uploads/). Using default vocabulary.")
        default_tokens = ['function', 'return', 'contract', 'uint', '{', '}', ';']
        vocab.update(default_tokens)

    # Explicitly add number tokens to prevent KeyError: '23'
    for i in range(0, 100):  # Adjust range as needed
        vocab.add(str(i))

    return vocab

def build_vocab_from_file():
    """Load vocabulary from vocab.txt file, improving robustness."""
    if not os.path.exists(VOCAB_FILE):
        print(f"Vocabulary file not found: {VOCAB_FILE}. Using default vocabulary.")
        vocab = build_vocab_from_code()  # Dynamically generate if file doesn't exist
    else:
        with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
            vocab = set(f.read().split('\n'))
    vocab.add('\n')
    vocab.add(' ')
    return vocab

if __name__ == "__main__":
    vocab = build_vocab_from_code()
    os.makedirs(os.path.dirname(VOCAB_FILE), exist_ok=True)
    with open(VOCAB_FILE, 'w', encoding='utf-8') as f:
        for word in vocab:
            f.write(f'{word}\n')
    print(f"Vocabulary saved to: {VOCAB_FILE}")
    print(f"Vocabulary size: {len(vocab)}")