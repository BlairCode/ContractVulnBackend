import torch
import torch.nn as nn
import random

class Sum(nn.Module):
    """
    Sum over a dimension. In this case we sum over lstm's sequence length dimension to combine information.
    """
    def __init__(self, dim, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.sum(x, dim=self.dim, keepdim=self.keepdim)

def generate_vector(start, step=1, size=3, noise=0):
    """
    Generates a vector starting from 'start' with a fixed 'step'.
    Optionally adds 'noise' to each element.
    
    Parameters:
        start (int): Starting value of the vector.
        step (int): Step size between elements.
        size (int): Number of elements in the vector.
        noise (int): Maximum noise to add/subtract from each element.
    
    Returns:
        list: Generated vector.
    """
    vector = [start + i * step for i in range(size)]
    if noise > 0:
        vector = [x + random.randint(-noise, noise) for x in vector]
    return vector

def generate_data(num_samples=30, noise=2, label0_ratio=0.7):
    """
    Generates a dataset based on the specified criteria, ensuring that label 0 is more frequent than label 1.
    
    Parameters:
        num_samples (int): Total number of data points to generate.
        noise (int): Maximum noise to add/subtract from each element.
        label0_ratio (float): Proportion of samples with label 0 (between 0 and 1).
    
    Returns:
        list: Generated dataset.
    """
    data = []
    labels = [0, 1]
    weights = [label0_ratio, 1 - label0_ratio]  # e.g., [0.7, 0.3] for 70% label 0 and 30% label 1

    for _ in range(num_samples):
        label = random.choices(labels, weights=weights, k=1)[0]
        
        if label == 0:
            # Decide randomly whether to increase or decrease
            direction = random.choice(['increase', 'decrease'])
            if direction == 'increase':
                start1 = random.randint(1, 10)
                start2 = random.randint(11, 20)
                vec1 = generate_vector(start1, step=random.randint(1,3), noise=noise)
                vec2 = generate_vector(start2, step=random.randint(1,3), noise=noise)
            else:
                start1 = random.randint(10, 20)
                start2 = random.randint(21, 30)
                vec1 = generate_vector(start1, step=-random.randint(1,3), noise=noise)
                vec2 = generate_vector(start2, step=-random.randint(1,3), noise=noise)
        else:
            # One increases, the other decreases
            direction = random.choice(['increase_decrease', 'decrease_increase'])
            if direction == 'increase_decrease':
                start_inc = random.randint(1, 10)
                start_dec = random.randint(10, 20)
                vec1 = generate_vector(start_inc, step=random.randint(1,3), noise=noise)
                vec2 = generate_vector(start_dec, step=-random.randint(1,3), noise=noise)
            else:
                start_dec = random.randint(10, 20)
                start_inc = random.randint(1, 10)
                vec1 = generate_vector(start_dec, step=-random.randint(1,3), noise=noise)
                vec2 = generate_vector(start_inc, step=random.randint(1,3), noise=noise)
        
        data.append([label, [vec1, vec2]])
    
    return data
