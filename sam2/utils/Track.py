#Track VRAM
def get_gpu_mem(device=None):
    """
    Returns current GPU memory usage (allocated) in MB.
    Accurate and efficient.
    """
    if not torch.cuda.is_available():
        return 0.0

    if device is None:
        device = torch.cuda.current_device()

    torch.cuda.synchronize(device)  # ensures accurate timing

    stats = torch.cuda.memory_stats(device)
    allocated = stats["allocated_bytes.all.current"] / 1024**2

    return allocated
