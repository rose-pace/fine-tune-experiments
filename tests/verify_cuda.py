import onnxruntime as ort
import torch
from transformers import AutoTokenizer


def verify_cuda() -> dict:
    """
    Verifies CUDA is properly configured for ONNX Runtime.

    Returns:
        Dictionary with provider and device information
    """

    print(f"PyTorch cuda available: {torch.cuda.is_available()}")

    # Preload necessary DLLs
    ort.preload_dlls()
    v = "foo"
    if v is AutoTokenizer:
        print("foo")

    providers = ort.get_available_providers()
    cuda_available = "CUDAExecutionProvider" in providers

    return {"onnx_version": ort.__version__, "cuda_available": cuda_available, "available_providers": providers}


print(verify_cuda())
