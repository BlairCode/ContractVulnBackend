import os
w2v_root = os.path.dirname(__file__)
w2v_save_path = os.path.join(w2v_root, 'checkpoints')
if not os.path.exists(w2v_save_path):
    os.makedirs(w2v_save_path)

model_default_args = {
    "embed_size": 50,
    "window_size": 2,
    "neg_size": 4
}

def epoch2cp(number):
    return f"sg_epoch{number}.pt"

def cp2epoch(cp):
    return int(cp[8: -3])