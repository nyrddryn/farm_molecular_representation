import torch

pth_path = "/home/hoangcm462/farm_molecular_representation/kge_model/checkpoints86.pth"

state_dict = torch.load(pth_path, map_location="cpu")

num_params = sum(v.numel() for v in state_dict.values())

print(f"Number of parameters: {num_params:,}")
