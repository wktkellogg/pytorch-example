import torch

def get_device():
    device = None
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)
        total_devices = torch.cuda.device_count()
        print(f"Current CUDA device index: {device}")
        print(f"Current CUDA device name: {device_name}")
        print(f"Total number of available CUDA devices: {total_devices}")
    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU.")
    return device

if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")
    x = torch.ones(1, device=device)
    print (x)
