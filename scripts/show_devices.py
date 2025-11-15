"""
List all available PyTorch devices
"""

import torch


def list_all_devices():
    """List all available PyTorch devices."""
    devices = []

    print("Available PyTorch Devices:")
    print("-" * 50)

    # CPU is always available
    devices.append("cpu")
    print(f"  ✅ cpu")

    # Check CUDA devices
    if torch.cuda.is_available():
        num_cuda = torch.cuda.device_count()
        for i in range(num_cuda):
            device_name = torch.cuda.get_device_name(i)
            devices.append(f"cuda:{i}")
            print(f"  ✅ cuda:{i} - {device_name}")

    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append("mps")
        print(f"  ✅ mps - Apple Silicon GPU")

    # Check XPU (Intel)
    if hasattr(torch, 'xpu') and hasattr(torch.xpu, 'is_available') and torch.xpu.is_available():
        num_xpu = torch.xpu.device_count()
        for i in range(num_xpu):
            device_name = torch.xpu.get_device_name(i)
            devices.append(f"xpu:{i}")
            print(f"  ✅ xpu:{i} - {device_name}")

    print("-" * 50)
    print(f"Total devices found: {len(devices)}")

    return devices


if __name__ == "__main__":
    list_all_devices()