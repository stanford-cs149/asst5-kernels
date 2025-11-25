import os
from pathlib import Path
import sys

def create_submission_cu():
    """
    Creates submission_cu.py by reading submission.cu and wrapping it with inline CUDA compilation.
    """
    
    # Read CUDA source code from submission.cu file
    cuda_source_path = 'submission.cu'
    
    if not os.path.exists(cuda_source_path):
        raise FileNotFoundError(f"Could not find {cuda_source_path}")
    
    cuda_source_escaped = Path(cuda_source_path).read_text()
    assert len(sys.argv) > 1, "Please provide SUNet ID as an argument"
    kid = sys.argv[1]
    
    # Create the submission_cu.py content
    submission_cu_content = f'''import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
import sys
import io

# CUDA source code loaded from submission.cu
cuda_source = """
{cuda_source_escaped}
"""

# C++ header declaration
cpp_source = """
#include <torch/extension.h>
torch::Tensor custom_kernel(
    torch::Tensor queries,
    torch::Tensor latents,
    torch::Tensor query_in_in_layer_weight,
    torch::Tensor query_in_in_layer_bias,
    torch::Tensor query_in_out_layer_weight,
    torch::Tensor query_in_out_layer_bias,
    torch::Tensor attn_c_q_weight,
    torch::Tensor attn_c_q_bias,
    torch::Tensor attn_c_k_weight,
    torch::Tensor attn_c_k_bias,
    torch::Tensor attn_c_v_weight,
    torch::Tensor attn_c_v_bias,
    torch::Tensor attn_c_proj_weight,
    torch::Tensor attn_c_proj_bias,
    torch::Tensor out_proj_weight,
    torch::Tensor out_proj_bias
);
"""

# Ensure stdout and stderr exist
if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

cuda_module = load_inline(
    name='submission_cuda_1d_occupancy_decoder_{kid}',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['custom_kernel'],
    verbose=True,  # Enable verbose to see compilation details
    # with_cuda=True,
    # build_directory=".",
)

def custom_kernel(data: input_t) -> output_t:
    # Occupancy decoder input_t: (queries, latents, weights)
    queries, latents, weights = data
    # For CUDA, we need to pass all weights as separate arguments
    # Extract weights from dictionary
    return cuda_module.custom_kernel(
        queries, latents,
        weights['query_in_in_layer_weight'],
        weights['query_in_in_layer_bias'],
        weights['query_in_out_layer_weight'],
        weights['query_in_out_layer_bias'],
        weights['attn_c_q_weight'],
        weights['attn_c_q_bias'],
        weights['attn_c_k_weight'],
        weights['attn_c_k_bias'],
        weights['attn_c_v_weight'],
        weights['attn_c_v_bias'],
        weights['attn_c_proj_weight'],
        weights['attn_c_proj_bias'],
        weights['out_proj_weight'],
        weights['out_proj_bias'],
    )
'''
    
    # Write to submission_cu.py
    output_path = 'submission.py'
    with open(output_path, 'w') as f:
        f.write(submission_cu_content)
    
    print(f"✓ Created {output_path} successfully!")
    print(f"  Read CUDA source from: {cuda_source_path}")


if __name__ == "__main__":
    create_submission_cu()