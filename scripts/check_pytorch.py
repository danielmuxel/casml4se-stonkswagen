"""
PyTorch System Information Checker
Displays PyTorch version, available hardware accelerators, and system resources.
Compatible with: macOS (Apple Silicon & Intel), Linux/Windows (NVIDIA CUDA), AMD ROCm
"""

import sys
import platform
import os


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f" {title}")
    print(f"{'=' * 70}")


def get_system_info():
    """Get basic system information."""
    info = {
        'os': platform.system(),
        'os_version': platform.release(),
        'platform': platform.platform(),
        'architecture': platform.machine(),
        'processor': platform.processor() or 'Unknown',
    }

    # macOS specific info
    if platform.system() == 'Darwin':
        try:
            mac_ver = platform.mac_ver()[0]
            if mac_ver:
                info['macos_version'] = mac_ver
        except:
            pass

    return info


def check_pytorch():
    """Check and display PyTorch installation and system information."""

    # Python Information
    print_section("Python Information")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")

    # System Information
    print_section("System Information")
    sys_info = get_system_info()
    print(f"Operating System: {sys_info['os']} {sys_info['os_version']}")
    if 'macos_version' in sys_info:
        print(f"macOS Version: {sys_info['macos_version']}")
    print(f"Platform: {sys_info['platform']}")
    print(f"Architecture: {sys_info['architecture']}")
    print(f"Processor: {sys_info['processor']}")

    # Detect Apple Silicon
    is_apple_silicon = (platform.system() == 'Darwin' and
                        platform.machine() == 'arm64')
    if is_apple_silicon:
        print("✅ Apple Silicon (M1/M2/M3) detected")

    # Try to import PyTorch
    try:
        import torch
        pytorch_available = True
    except ImportError:
        print_section("PyTorch Status")
        print("❌ PyTorch is not installed!")
        print("Install it with:")
        print("  - For CUDA: pip install torch")
        print("  - For ROCm: pip install torch --index-url https://download.pytorch.org/whl/rocm6.1")
        print("  - For macOS: pip install torch")
        return

    # PyTorch Version Information
    print_section("PyTorch Information")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"PyTorch File Location: {torch.__file__}")
    print(f"PyTorch Build: ", end="")

    # Determine build type
    build_types = []
    if hasattr(torch.version, 'cuda') and torch.version.cuda:
        build_types.append(f"CUDA {torch.version.cuda}")
    if hasattr(torch.version, 'hip') and torch.version.hip:
        build_types.append(f"ROCm/HIP {torch.version.hip}")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_built():
        build_types.append("MPS")
    if not build_types:
        build_types.append("CPU-only")
    print(", ".join(build_types))

    # CPU Information
    print_section("CPU Information")
    cpu_count = os.cpu_count()
    print(f"CPU Count (Logical): {cpu_count}")
    print(f"PyTorch Threads: {torch.get_num_threads()}")

    # Check for optimized CPU libraries
    if hasattr(torch.backends, 'mkl'):
        mkl_available = torch.backends.mkl.is_available()
        print(f"MKL (Intel Math Kernel Library): {'✅ Available' if mkl_available else '❌ Not available'}")

    if hasattr(torch.backends, 'openmp'):
        openmp_available = torch.backends.openmp.is_available()
        print(f"OpenMP: {'✅ Available' if openmp_available else '❌ Not available'}")

    # CUDA/NVIDIA GPU Information
    print_section("NVIDIA GPU Information (CUDA)")
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        print(f"✅ CUDA Available: Yes")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")

        if hasattr(torch.backends, 'cudnn'):
            print(f"cuDNN Available: {torch.backends.cudnn.is_available()}")
            if torch.backends.cudnn.is_available():
                print(f"cuDNN Version: {torch.backends.cudnn.version()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\n  GPU {i}: {props.name}")
            print(f"    Compute Capability: {props.major}.{props.minor}")
            print(f"    Multi-Processor Count: {props.multi_processor_count}")

            # Memory Information
            try:
                mem_allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
                mem_reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
                mem_total = props.total_memory / 1024 ** 3

                print(f"    Total Memory: {mem_total:.2f} GB")
                print(f"    Allocated Memory: {mem_allocated:.2f} GB")
                print(f"    Reserved Memory: {mem_reserved:.2f} GB")
                print(f"    Free Memory: {mem_total - mem_reserved:.2f} GB")
            except Exception as e:
                print(f"    Error getting memory info: {e}")
    else:
        print("❌ CUDA Available: No")
        if platform.system() == 'Darwin':
            print("   (Expected on macOS - use MPS instead)")
        else:
            print("   (Either no NVIDIA GPU or CUDA not installed)")

    # AMD GPU (ROCm) Information
    print_section("AMD GPU Information (ROCm)")
    has_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None

    if has_rocm:
        print(f"✅ ROCm/HIP Available: Yes")
        print(f"ROCm/HIP Version: {torch.version.hip}")

        # Try to get AMD GPU info
        if torch.cuda.is_available():  # ROCm uses the cuda API
            print(f"AMD GPU Device Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                try:
                    props = torch.cuda.get_device_properties(i)
                    mem_total = props.total_memory / 1024 ** 3
                    print(f"    Total Memory: {mem_total:.2f} GB")
                except:
                    pass
    else:
        print("❌ ROCm/HIP Available: No")
        if not platform.system() == 'Darwin':
            print("   (Install ROCm-enabled PyTorch for AMD GPU support)")

    # Apple Silicon (MPS) Information
    print_section("Apple Silicon Information (MPS)")
    has_mps = hasattr(torch.backends, 'mps')

    if has_mps:
        mps_built = torch.backends.mps.is_built()
        mps_available = torch.backends.mps.is_available() if mps_built else False

        if mps_available:
            print("✅ Apple MPS (Metal Performance Shaders): Available")
            print("   PyTorch can use Apple Silicon GPU")

            # Try to get more info
            if is_apple_silicon:
                print(f"   Architecture: {platform.machine()}")
        elif mps_built:
            print("⚠️  Apple MPS: Built but not available")
            print("   (May require newer macOS version)")
        else:
            print("❌ Apple MPS: Not built in this PyTorch version")
    else:
        print("❌ Apple MPS: Not supported")
        if is_apple_silicon:
            print("   (Upgrade PyTorch to version with MPS support)")

    # Intel XPU/NPU Information
    print_section("Intel Accelerator Information (XPU)")
    if hasattr(torch, 'xpu') and hasattr(torch.xpu, 'is_available'):
        if torch.xpu.is_available():
            print("✅ Intel XPU Available: Yes")
            print(f"XPU Device Count: {torch.xpu.device_count()}")
            for i in range(torch.xpu.device_count()):
                print(f"  XPU {i}: {torch.xpu.get_device_name(i)}")
        else:
            print("❌ Intel XPU Available: No (hardware not detected)")
    else:
        print("❌ Intel XPU: Not supported in this PyTorch build")

    # System Memory and Disk Information
    print_section("System Resources")
    try:
        import psutil

        # Memory
        mem = psutil.virtual_memory()
        print("RAM:")
        print(f"  Total: {mem.total / 1024 ** 3:.2f} GB")
        print(f"  Available: {mem.available / 1024 ** 3:.2f} GB")
        print(f"  Used: {mem.used / 1024 ** 3:.2f} GB ({mem.percent}%)")
        print(f"  Free: {mem.free / 1024 ** 3:.2f} GB")

        # Swap
        swap = psutil.swap_memory()
        print(f"\nSwap:")
        print(f"  Total: {swap.total / 1024 ** 3:.2f} GB")
        print(f"  Used: {swap.used / 1024 ** 3:.2f} GB ({swap.percent}%)")
        print(f"  Free: {swap.free / 1024 ** 3:.2f} GB")

        # Disk
        disk = psutil.disk_usage('/')
        print(f"\nDisk (Root):")
        print(f"  Total: {disk.total / 1024 ** 3:.2f} GB")
        print(f"  Used: {disk.used / 1024 ** 3:.2f} GB ({disk.percent}%)")
        print(f"  Free: {disk.free / 1024 ** 3:.2f} GB")

        # CPU Usage
        cpu_percent = psutil.cpu_percent(interval=1, percpu=False)
        print(f"\nCPU Usage: {cpu_percent}%")

    except ImportError:
        print("⚠️  psutil not installed")
        print("Install with: pip install psutil")
        print("(System resource information not available)")

    # PyTorch Modules/Submodules
    print_section("PyTorch Modules")

    key_modules = {
        'nn': 'Neural Network building blocks',
        'optim': 'Optimization algorithms',
        'utils': 'Utility functions',
        'autograd': 'Automatic differentiation',
        'cuda': 'CUDA utilities',
        'distributed': 'Distributed training',
        'jit': 'TorchScript JIT compiler',
        'onnx': 'ONNX export support',
        'quantization': 'Model quantization',
        'sparse': 'Sparse tensor support',
    }

    print("Key PyTorch Submodules:")
    for module, description in key_modules.items():
        if hasattr(torch, module):
            print(f"  ✅ torch.{module:<15} - {description}")
        else:
            print(f"  ❌ torch.{module:<15} - {description}")

    # Additional Configuration
    print_section("PyTorch Configuration")
    print(f"Deterministic Algorithms: {torch.are_deterministic_algorithms_enabled()}")
    print(f"Default Tensor Type: {torch.get_default_dtype()}")
    print(f"Gradient Enabled: {torch.is_grad_enabled()}")

    # Determine best available device
    print_section("Recommended Device")
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        print(f"✅ Recommended: '{device}' ({device_name})")
    elif has_mps and torch.backends.mps.is_available():
        device = "mps"
        print(f"✅ Recommended: '{device}' (Apple Silicon GPU)")
    else:
        device = "cpu"
        print(f"✅ Recommended: '{device}' (CPU-only)")

    print(f"\nUse in code: device = torch.device('{device}')")

    # Quick Functionality Test
    print_section("Functionality Test")
    try:
        # CPU tensor
        cpu_tensor = torch.randn(3, 3)
        print("✅ CPU tensor creation: Success")

        # CUDA tensor
        if torch.cuda.is_available():
            try:
                gpu_tensor = torch.randn(3, 3, device='cuda')
                result = (gpu_tensor @ gpu_tensor.T).cpu()
                print("✅ CUDA tensor creation and computation: Success")
            except Exception as e:
                print(f"❌ CUDA test failed: {e}")

        # MPS tensor
        if has_mps and torch.backends.mps.is_available():
            try:
                mps_tensor = torch.randn(3, 3, device='mps')
                result = (mps_tensor @ mps_tensor.T).cpu()
                print("✅ MPS tensor creation and computation: Success")
            except Exception as e:
                print(f"⚠️  MPS test failed: {e}")

    except Exception as e:
        print(f"❌ Error in functionality test: {e}")

    print("\n" + "=" * 70)
    print("System check complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    check_pytorch()