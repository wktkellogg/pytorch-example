import torch
from rich.console import Console
from rich.text import Text
import click


console = Console()

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

@click.group()
def cli():
    """A simple CLI application using Click."""
    pass

@cli.command()
def show_device():
    """Displays the current device being used (CPU or GPU)."""
    device = get_device()
    if device.type == 'cuda':
        text = Text(f"Using CUDA device: {device}", style="bold green")
    elif device.type == 'mps':
        text = Text(f"Using Metal Performance Shaders (MPS) device: {device}", style="bold green")
    else:
        text = Text("Using CPU", style="bold red")
    console.print(text)
    console.print(torch.ones(1, device=device))

# Run the CLI
if __name__ == "__main__":
    cli()
