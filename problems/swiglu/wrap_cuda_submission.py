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
    torch::Tensor x,
    torch::Tensor W,
    torch::Tensor V,
    torch::Tensor b,
    torch::Tensor c,
    float beta
);
"""

# Ensure stdout and stderr exist
if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

cuda_module = load_inline(
    name='submission_cuda_swiglu_{kid}',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['custom_kernel'],
    verbose=True,  # Enable verbose to see compilation details
    # with_cuda=True,
    # build_directory=".",
)

def custom_kernel(data: input_t) -> output_t:
    # SwiGLU input_t: (x, W, V, b, c, beta)
    x, W, V, b, c, beta = data
    return cuda_module.custom_kernel(
        x,
        W,
        V,
        b,
        c,
        beta,
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