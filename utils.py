import torch.cuda


def dev():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")