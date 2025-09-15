import torch

print(f"torch: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Cuda available?: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")