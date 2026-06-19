import nbformat, sys
nb = nbformat.read("notebooks/00_Introduction.ipynb", as_version=4)
errors, code_cells, with_output = [], 0, 0
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    code_cells += 1
    has_out = False
    for o in c.get("outputs", []):
        if o.output_type == "error":
            errors.append((i, o.ename, o.evalue))
        if o.output_type in ("stream", "execute_result", "display_data"):
            has_out = True
    if has_out:
        with_output += 1
print(f"code cells: {code_cells}, produced output: {with_output}, errors: {len(errors)}")
for e in errors:
    print("  ERROR cell", e)

# Spot-check a couple of key hand-calculated results appear in outputs
joined = "\n".join(
    "".join(o.get("text", "") for o in c.get("outputs", []) if o.output_type == "stream")
    for c in nb.cells if c.cell_type == "code"
)
checks = {
    "dot product = 8": "8.0" in joined,
    "W@x = [4,1,7]": "[4.0, 1.0, 7.0]" in joined,
    "z = [5,2,8]": "[5.0, 2.0, 8.0]" in joined,
    "softmax 0.665": "0.665" in joined,
    "attention out 0.802": "0.802" in joined,
    "loss 0.408": "0.408" in joined,
    "grad descent -> 3.000": "3.000" in joined,
}
print("\nhand-vs-code spot checks:")
for k, v in checks.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")
sys.exit(1 if errors or not all(checks.values()) else 0)
