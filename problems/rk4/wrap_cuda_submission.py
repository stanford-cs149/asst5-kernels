import os
from pathlib import Path
import sys

def create_submission_cu():
    """
    Creates submission.py by reading submission.cu and wrapping it with inline CUDA compilation.
    """
    
    # Read CUDA source code from submission.cu file
    cuda_source_path = 'submission.cu'
    
    if not os.path.exists(cuda_source_path):
        raise FileNotFoundError(f"Could not find {cuda_source_path}")
    
    cuda_source_escaped = Path(cuda_source_path).read_text()

    assert len(sys.argv) > 1, "Please provide SUNet ID as an argument"
    kid = sys.argv[1]
    # Create the submission.py content
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
torch::Tensor custom_kernel(torch::Tensor u0,
            float alpha,
            float hx,
            float hy,
            float hz,
            int n_steps);
"""

# Ensure stdout and stderr exist
if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

try:
    cuda_module = load_inline(
        name='submission_cuda_rk4_{kid}',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['custom_kernel'],
        verbose=True,
        # with_cuda=True,
        # build_directory=".",  # Cache compiled modules here
    )
except Exception as e:
    print("\\n====== CUDA Extension Build Error ======\\n")
    print(e)                     # short error
    print("\\n====== Full Traceback ======\\n")
    import traceback
    traceback.print_exc()        # full Python traceback
    raise  

def custom_kernel(data: input_t) -> output_t:
    # RK4 assignment input_t: (u0, alpha, hx, hy, hz, n_steps)
    u0, alpha, hx, hy, hz, n_steps = data
    return cuda_module.custom_kernel(
        u0,
        alpha,
        hx,
        hy,
        hz,
        n_steps,
    )
'''
    
    # Write to submission.py
    output_path = 'submission.py'
    with open(output_path, 'w') as f:
        f.write(submission_cu_content)
    
    print(f"✓ Created {output_path} successfully!")
    print(f"  Read CUDA source from: {cuda_source_path}")


if __name__ == "__main__":
    create_submission_cu()
