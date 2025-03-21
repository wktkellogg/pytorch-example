import torch

def check_mps_availability():
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print (x)
    else:
        print ("MPS device not found.")

def check_cuda_availability():
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available!")
        
        # Get the current device
        current_device = torch.cuda.current_device()
        
        # Get the name of the current device
        device_name = torch.cuda.get_device_name(current_device)
        
        # Get the total number of CUDA devices
        total_devices = torch.cuda.device_count()
        
        print(f"Current CUDA device index: {current_device}")
        print(f"Current CUDA device name: {device_name}")
        print(f"Total number of available CUDA devices: {total_devices}")
    else:
        print("CUDA is not available. Using CPU instead.")

if __name__ == "__main__":
    check_mps_availability()
    check_cuda_availability()
