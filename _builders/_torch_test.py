import os, sys
print("python:", sys.version)
print("KMP_DUPLICATE_LIB_OK =", os.environ.get("KMP_DUPLICATE_LIB_OK"))
try:
    import torch
    print("torch OK", torch.__version__)
    print("tiny op:", (torch.ones(2) + 1).tolist())
except Exception as e:
    print("FAIL", type(e).__name__, e)
