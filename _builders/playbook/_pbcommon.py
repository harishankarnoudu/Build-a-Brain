"""
Shared helpers for building the Playbook notebooks.

Design goals (match the project philosophy in build_00.py):
  * Generate notebooks with nbformat so we never hand-escape JSON.
  * Markdown uses plain-text math (×, →, sqrt(), exp()) so a struggling reader
    never fights LaTeX — every formula is also worked by hand in the text and
    then verified by a code cell directly below it.
  * `readonly()` emits a NON-executed code cell, used for TensorFlow / Keras /
    HuggingFace snippets that cannot run on this locked-down machine. They are
    clearly labelled so the learner knows to read, not run, them.

Usage in a builder:
    from _pbcommon import md, code, readonly, build, reset
    reset()
    md("# title ...")
    code("import numpy as np ...")
    build("P00_Python_and_NumPy")        # writes playbook/P00_*.ipynb (no execute)
"""
import os
import subprocess
import sys
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

# project root = two levels up from this file (_builders/playbook/_pbcommon.py)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLAYBOOK_DIR = os.path.join(ROOT, "playbook")

_cells = []


def reset():
    """Start a fresh notebook (call once at the top of each builder)."""
    _cells.clear()


def md(text):
    """Add a markdown cell."""
    _cells.append(new_markdown_cell(text.strip("\n")))


def code(text):
    """Add a runnable code cell."""
    _cells.append(new_code_cell(text.strip("\n")))


def readonly(text, note="READ-ONLY — illustrative library code, not executed here"):
    """Add a NON-runnable code cell (TF/Keras/HuggingFace).

    We tag it so nbconvert skips execution, and prepend a banner comment so the
    reader knows this is for understanding the API, not for running.
    """
    banner = f"# === {note} ===\n"
    cell = new_code_cell(banner + text.strip("\n"))
    cell.metadata["tags"] = ["skip-execution"]
    _cells.append(cell)


def build(name, execute=False, kernel="tinygpt"):
    """Write playbook/<name>.ipynb. If execute=True, run it in place via nbconvert.

    Returns the absolute path to the notebook.
    """
    os.makedirs(PLAYBOOK_DIR, exist_ok=True)
    nb = new_notebook(cells=list(_cells))
    nb.metadata["kernelspec"] = {
        "display_name": "Python (TinyGPT)",
        "language": "python",
        "name": kernel,
    }
    path = os.path.join(PLAYBOOK_DIR, name + ".ipynb")
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", path, f"({len(_cells)} cells)")

    if execute:
        cmd = [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            "--ExecutePreprocessor.timeout=300",
            f"--ExecutePreprocessor.kernel_name={kernel}",
            path,
        ]
        print("executing:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print("executed OK", path)
    return path
