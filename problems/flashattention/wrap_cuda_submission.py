import os
from pathlib import Path
import sys

def create_submission():
    """
    Creates submission.py by reading submission.cu and wrapping it
    with inline CUDA compilation logic via torch.utils.cpp_extension.load_inline.
    """

    # 1. Read CUDA source code
    cuda_source_path = "submission.cu"

    if not os.path.exists(cuda_source_path):
        raise FileNotFoundError(f"Could not find {cuda_source_path}")

    # Read text
    raw_cuda_source = Path(cuda_source_path).read_text()
    assert len(sys.argv) > 1, "Please provide SUNet ID as an argument"
    kid = sys.argv[1]

    # 2. Define the content for the new submission.py file
    submission_content = f'''import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
import sys
import io

# ------------------------------------------------------
# 1. CUDA Source Code (Injected automatically)
# ------------------------------------------------------
cuda_source = """
{raw_cuda_source}
"""

# ------------------------------------------------------
# 2. C++ Header Declaration
# ------------------------------------------------------
# This signature must match the C++ function in the .cu file
cpp_source = """
#include <torch/extension.h>
torch::Tensor flash_attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
"""

# ------------------------------------------------------
# 3. JIT Compilation
# ------------------------------------------------------
# Ensure stdout/stderr exist to prevent load_inline issues in some environments (e.g. Colab/Jupyter)
if sys.stdout is None: sys.stdout = io.StringIO()
if sys.stderr is None: sys.stderr = io.StringIO()

# print("Compiling FlashAttention CUDA kernel... (This may take a moment)")

cuda_module = load_inline(
    name='flash_attention_cuda_{kid}',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['flash_attention_forward'],
    verbose=True,        # Print compilation log
    # with_cuda=True,
    # extra_cuda_cflags=["-O2"]
)
# print("Compilation complete.")

# ------------------------------------------------------
# 4. Python Wrapper
# ------------------------------------------------------
def custom_kernel(data: input_t) -> output_t:
    """
    Wrapper function to call the compiled CUDA kernel.

    Args:
        data: tuple of (Q, K, V)
            Q: (batch_size, num_heads, seq_len, head_dim)
            K: (batch_size, num_heads, seq_len, head_dim)
            V: (batch_size, num_heads, seq_len, head_dim)
    Returns:
        Output tensor of shape (batch_size, num_heads, seq_len, head_dim)
    """
    q, k, v = data
    # Ensure inputs are fp16 and on CUDA
    q = q.half().cuda()
    k = k.half().cuda()
    v = v.half().cuda()

    return cuda_module.flash_attention_forward(q, k, v)

'''

    # 3. Write the generated content to submission.py
    output_path = "submission.py"
    with open(output_path, "w") as f:
        f.write(submission_content)

    print(f"✓ Created {output_path} successfully!")
    print(f"  Read CUDA source from: {cuda_source_path}")


if __name__ == "__main__":
    create_submission()
