import torch

x = torch.load("cat.0__left_ear__2__acts.pt", map_location="cpu")

for k, v in x.items():
    print(f"{k:7s} -> {v.shape}")
