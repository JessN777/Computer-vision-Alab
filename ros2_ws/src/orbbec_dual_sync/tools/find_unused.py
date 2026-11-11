"""Find defs in orbbec_dual_sync that nothing else names."""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "orbbec_dual_sync"
ROOTS = [
    PKG,
    PKG.parent / "test",
    Path(r"C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics"),
    Path(r"C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\ros2_ws\src\orbbec_dual_sync"),
]


def py_files(root: Path):
    if not root.exists():
        return
    yield from root.rglob("*.py")


def collect_defs(tree: ast.AST, path: str):
    out = []

    class V(ast.NodeVisitor):
        def __init__(self):
            self.stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef):
            out.append((node.name, path, node.lineno, "class", ".".join(self.stack)))
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef):
            kind = "method" if self.stack else "func"
            out.append(
                (node.name, path, node.lineno, kind, ".".join(self.stack))
            )
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

    V().visit(tree)
    return out


def main() -> None:
    defs: dict[str, list[tuple]] = defaultdict(list)
    for p in py_files(PKG):
        src = p.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(src)
        except SyntaxError as exc:
            print("SYNTAX", p, exc)
            continue
        for name, path, lineno, kind, parent in collect_defs(tree, str(p)):
            defs[name].append((path, lineno, kind, parent))

    counts: dict[str, int] = defaultdict(int)
    for root in ROOTS:
        for p in py_files(root):
            if p.name == "find_unused.py":
                continue
            src = p.read_text(encoding="utf-8", errors="replace")
            for name in list(defs):
                if name in src:
                    counts[name] += src.count(name)

    skip = {
        "main",
        "__init__",
        "__enter__",
        "__exit__",
        "__repr__",
        "__str__",
        "__call__",
        "__iter__",
        "__len__",
        "__getitem__",
        "__setitem__",
        "__contains__",
        "__bool__",
        "__eq__",
        "__hash__",
        "__getattr__",
        "__setattr__",
    }
    rows = []
    for name, locs in defs.items():
        if name in skip:
            continue
        nloc = len(locs)
        nref = counts.get(name, 0)
        rows.append((nref - nloc, nref, nloc, name, locs))
    rows.sort()
    print("LOWEST EXTRA REFS (nref - ndefs). 0/negative => likely unused")
    for extra, nref, nloc, name, locs in rows[:60]:
        print(f"\n{name}  extra={extra} refs={nref} defs={nloc}")
        for path, lineno, kind, parent in locs[:3]:
            q = f"{parent}.{name}" if parent else name
            print(f"  {kind} {q} @ {Path(path).name}:{lineno}")


if __name__ == "__main__":
    main()
