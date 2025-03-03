import os
model_root = os.path.dirname(__file__)
model_save_path = os.path.join(model_root, 'checkpoints')
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
blstm_default_args = {
    'hidden_size': 256,
    'dropout': 0.1
}

# 'lr': 0.0001,
# 'epochs': 2000,

def epoch2cp(number):
    return f"blstm_epoch{number}.pt"

def cp2epoch(cp):
    return int(cp[11:-3])