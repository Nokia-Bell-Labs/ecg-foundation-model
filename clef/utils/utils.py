import random
import numpy as np
import torch
import time
from functools import wraps

CHANNEL_MAPPING = {
    "I": 0,
    "II": 1,
    "III": 2,
    "AVR": 3,
    "AVL": 4,
    "AVF": 5,
    "V1": 6,
    "V2": 7,
    "V3": 8,
    "V4": 9,
    "V5": 10,
    "V6": 11,
    "X": 12,
    "Y": 13,
    "Z": 14,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def timing(enabled=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled:
                return func(*args, **kwargs)
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"{func.__name__} took {end - start:.2f} seconds")
            return result

        return wrapper

    return decorator


# if the signal is too flat, return False
def is_flat_signal(signal, threshold=0.01):
    if len(signal) == 0:
        return True
    signal = np.array(signal)
    return np.std(signal) < threshold


# if the signal contains too many 0 or nans, return False
def is_valid_signal(signal, zero_threshold=0.1, nan_threshold=0.1):
    if len(signal) == 0:
        return False
    signal = np.array(signal)
    num_zeros = np.sum(signal == 0)
    num_nans = np.sum(np.isnan(signal))
    total_length = len(signal)

    zero_ratio = num_zeros / total_length
    nan_ratio = num_nans / total_length

    return zero_ratio < zero_threshold and nan_ratio < nan_threshold


def load_records(file_path):
    with open(file_path, "r") as f:
        for line in f:
            yield line.strip()