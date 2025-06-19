# utils.py
import torch

def get_device() -> torch.device:
    """
    Dynamically selects the best available device (CUDA for NVIDIA, MPS for Apple Silicon, or CPU).
    """
    if torch.cuda.is_available():
        print("CUDA device found, using GPU.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("MPS device found, using Apple Silicon GPU.")
        return torch.device("mps")
    else:
        print("No GPU found, using CPU.")
        return torch.device("cpu")