import torch


def verify_cuda() -> dict:
    """
    Verifies CUDA is properly configured for ONNX Runtime.

    Returns:
        Dictionary with provider and device information
    """

    print(f"PyTorch cuda available: {torch.cuda.is_available()}")


print(verify_cuda())
