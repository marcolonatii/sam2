#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Build script for SAM 2 CUDA extensions.
This script handles the CUDA extension compilation that was previously in setup.py.
"""

import os
import sys
import subprocess
from pathlib import Path


def build_cuda_extension():
    """Build the SAM 2 CUDA extension."""
    
    # Environment variables for controlling build behavior
    BUILD_CUDA = os.getenv("SAM2_BUILD_CUDA", "1") == "1"
    BUILD_ALLOW_ERRORS = os.getenv("SAM2_BUILD_ALLOW_ERRORS", "1") == "1"
    
    if not BUILD_CUDA:
        print("CUDA extension build disabled via SAM2_BUILD_CUDA=0")
        return True
    
    # Error message template
    CUDA_ERROR_MSG = (
        "{}\n\n"
        "Failed to build the SAM 2 CUDA extension due to the error above. "
        "You can still use SAM 2 and it's OK to ignore the error above, although some "
        "post-processing functionality may be limited (which doesn't affect the results in most cases; "
        "(see https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).\n"
    )
    
    try:
        import torch
        from torch.utils.cpp_extension import load
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            if BUILD_ALLOW_ERRORS:
                print("CUDA not available, skipping CUDA extension build")
                return True
            else:
                raise RuntimeError("CUDA not available but CUDA extension build required")
        
        # Source files and compilation arguments
        source_dir = Path(__file__).parent / "sam2" / "csrc"
        sources = [str(source_dir / "connected_components.cu")]
        
        # Check if source files exist
        for src in sources:
            if not Path(src).exists():
                error_msg = f"Source file not found: {src}"
                if BUILD_ALLOW_ERRORS:
                    print(CUDA_ERROR_MSG.format(error_msg))
                    return True
                else:
                    raise FileNotFoundError(error_msg)
        
        extra_compile_args = {
            "cxx": [],
            "nvcc": [
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ],
        }
        
        print("Building SAM 2 CUDA extension...")
        
        # Build the extension
        sam2_cuda = load(
            name="sam2._C",
            sources=sources,
            extra_compile_args=extra_compile_args,
            verbose=True,
            with_cuda=True,
        )
        
        print("SAM 2 CUDA extension built successfully!")
        return True
        
    except Exception as e:
        error_msg = f"Error building CUDA extension: {e}"
        if BUILD_ALLOW_ERRORS:
            print(CUDA_ERROR_MSG.format(error_msg))
            return True
        else:
            print(error_msg, file=sys.stderr)
            return False


def main():
    """Main entry point for the build script."""
    success = build_cuda_extension()
    if not success:
        sys.exit(1)
    print("Build completed successfully!")


if __name__ == "__main__":
    main()
