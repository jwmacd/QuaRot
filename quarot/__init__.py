import torch
from . import nn
from . import functional


import quarot._CUDA


__all__ = [ 
           "matmul", #int-4 matmul
           "sym_quant", "sym_dequant", "PackedQuantizedTensor", # Quantization
           "apply_quarot_to_model",  # High-level API
]

class ShapeHandler:
    def __init__(self, x: torch.Tensor):
        self.size_excl_last = x.numel()//x.shape[-1]
        self.shape_excl_last = tuple(x.shape[:-1])

    # Keep the last dim unchanged, flatten all previous dims
    def flatten(self, x: torch.Tensor):
        return x.view(self.size_excl_last, -1)

    # Recover back to the original shape.
    def unflatten(self, x: torch.Tensor):
        return x.view(self.shape_excl_last + (-1,))

    def unflatten_scale(self, x: torch.Tensor):
        return x.view(self.shape_excl_last)


def flatten_last_dim_and_return_shape(x: torch.Tensor):
    shape_excl_last = x.shape[:-1]
    x = x.view(-1, x.shape[-1])
    return x, shape_excl_last


def matmul(A, B):
    assert A.shape[-1] % 32 == 0, "A.shape[-1]: {} must be multiplication of 32".format(A.shape[-1])
    A, A_shape_excl_last = flatten_last_dim_and_return_shape(A)
    B, B_shape_excl_last = flatten_last_dim_and_return_shape(B)
    return quarot._CUDA.matmul(A, B).view(*A_shape_excl_last, *B_shape_excl_last)

def sym_quant(x, scale):
    assert x.dtype == scale.dtype == torch.float16
    x, x_shape_excl_last = flatten_last_dim_and_return_shape(x)
    return quarot._CUDA.sym_quant(x, scale.view(-1)).view(*x_shape_excl_last, -1)

def sym_dequant(q, scale_row, scale_col, bits=32):
    assert q.dtype == torch.int32
    assert scale_row.dtype == scale_col.dtype == torch.float16
    q, q_shape_excl_last = flatten_last_dim_and_return_shape(q)
    return quarot._CUDA.sym_dequant(q, scale_row.view(-1), scale_col, bits).view(*q_shape_excl_last, -1)


class PackedQuantizedTensor:
    def __init__(self, 
                 quantized_x: torch.Tensor, 
                 scales_x: torch.Tensor):
        self.quantized_x = quantized_x
        self.scales_x = scales_x

    def size(self):
        return self.quantized_x.size()
    
    @property
    def device(self):
        return self.quantized_x.device
    
    @property
    def dtype(self):
        return self.quantized_x.dtype


def apply_quarot_to_model(model, mode='hadamard'):
    """Apply QuaRot rotation to a model - simplified API for external use.
    
    Args:
        model: The model to rotate (must be on CPU)
        mode: Rotation mode ('hadamard' or 'random')
    
    Returns:
        The rotated model (in-place modification)
    """
    import sys
    import os
    
    # Add fake_quant to path temporarily  
    quarot_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fake_quant_path = os.path.join(quarot_root, 'fake_quant')
    
    # Save current sys.path
    old_path = sys.path.copy()
    
    try:
        # Clear sys.path and add only what we need
        sys.path = [fake_quant_path, quarot_root] + [p for p in sys.path if '/app' not in p]
        
        # Import utils first to ensure it's available for rotation_utils
        sys.path.insert(0, fake_quant_path)
        import utils
        
        # Now import rotation_utils - it should find the utils module
        from fake_quant import rotation_utils
        
        # Mock args object 
        class Args:
            rotate_mode = mode
            fp32_had = False
            
        args = Args()
        
        # Apply rotation
        rotation_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, args)
        
    finally:
        # Restore sys.path
        sys.path = old_path
    
    return model
