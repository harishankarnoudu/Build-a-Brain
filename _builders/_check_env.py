import importlib
mods = ["matplotlib", "matplotlib.pyplot", "numpy", "pandas",
        "seaborn", "sklearn", "tqdm", "ipykernel", "jupyter_client",
        "nbconvert", "torch"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "")
        print(f"OK   {m} {ver}")
    except Exception as e:
        print(f"MISS {m} -> {type(e).__name__}: {e}")
