import os
import torch
import torch.distributed as dist


def setup_distributed():
    """Initialize distributed training environment.
    
    Must be called at the start of each training process when using multi-GPU training.
    Sets up the PyTorch distributed backend (NCCL for GPU).
    
    Returns:
        tuple: (rank, world_size, local_rank)
            - rank: Global rank (0 to world_size-1)
            - world_size: Total number of processes
            - local_rank: Local rank on this node (device index)
    """
    # Get environment variables set by torch.distributed.launch or torchrun
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Only initialize if world_size > 1 (multi-GPU mode)
    if world_size > 1:
        # Set the device
        torch.cuda.set_device(local_rank)
        
        # Initialize the process group
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )
        print(f"[distributed] rank={rank} world_size={world_size} local_rank={local_rank} initialized")
    else:
        print(f"[distributed] single GPU mode (world_size=1)")
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training environment.
    
    Must be called at the end of training to properly shutdown the distributed backend.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
        print("[distributed] cleanup complete")


def is_main_process():
    """Check if this is the main process (rank 0).
    
    Returns:
        bool: True if rank is 0, False otherwise.
    """
    if not torch.distributed.is_available():
        return True
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def get_rank():
    """Get the global rank of the current process.
    
    Returns:
        int: Global rank (0 to world_size-1).
    """
    if not torch.distributed.is_available():
        return 0
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def get_world_size():
    """Get the total number of processes in the distributed training job.
    
    Returns:
        int: World size (1 if not in distributed mode).
    """
    if not torch.distributed.is_available():
        return 1
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_local_rank():
    """Get the local rank on the current node.
    
    Returns:
        int: Local rank (device index on this node).
    """
    return int(os.environ.get("LOCAL_RANK", 0))


def synchronize():
    """Synchronize all processes (barrier).
    
    All processes will wait at this point until all others reach it.
    Only has effect in distributed mode.
    """
    if get_world_size() > 1:
        dist.barrier()


def reduce_dict(input_dict, average=True):
    """Reduce a dictionary of tensors across all ranks.
    
    Args:
        input_dict (dict): Dictionary with tensor values to reduce.
        average (bool): If True, average the values; if False, sum them.
    
    Returns:
        dict: Dictionary with reduced values. Only meaningful on rank 0.
    """
    world_size = get_world_size()
    
    if world_size == 1:
        return input_dict
    
    with torch.no_grad():
        # Convert all values to tensors and move to GPU
        tensor_dict = {}
        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                tensor_dict[k] = v.detach().clone()
            else:
                tensor_dict[k] = torch.tensor(float(v), device=f"cuda:{get_local_rank()}")
        
        # Reduce all tensors
        for k in tensor_dict:
            dist.all_reduce(tensor_dict[k], op=dist.ReduceOp.SUM)
            if average:
                tensor_dict[k] /= world_size
        
        # Convert back to Python scalars for logging
        output_dict = {}
        for k, v in tensor_dict.items():
            output_dict[k] = v.item()
        
        return output_dict


def reduce_loss(loss_dict):
    """Reduce loss values across all ranks (averaged).
    
    Convenience wrapper for reduce_dict with averaging enabled.
    
    Args:
        loss_dict (dict): Dictionary with loss values.
    
    Returns:
        dict: Dictionary with averaged loss values.
    """
    return reduce_dict(loss_dict, average=True)


def get_device(local_rank=None):
    """Get the appropriate device for the current process.
    
    Args:
        local_rank (int, optional): Local rank to use. If None, gets from environment.
    
    Returns:
        torch.device: CUDA device if available, CPU otherwise.
    """
    if local_rank is None:
        local_rank = get_local_rank()
    
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")
