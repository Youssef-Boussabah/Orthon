"""Repository source readers for test architecture fences."""
import ast
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[2]

def read_text(path):
    return (REPO_ROOT / path).read_text(encoding="utf-8")

def strip_cpp_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)

def read_cpp_source(path):
    # Resolve retained source-parameter spellings to their canonical files.
    path = str(path).replace("src/ops/cpu/", "src/ops/").replace("src/ops/cuda/", "src/backends/cuda/")
    return strip_cpp_comments(read_text(path))

def parse_python(path):
    return ast.parse(read_text(path), filename=str(path))

def iter_source_files(*roots, suffixes=(".h", ".hpp", ".cuh", ".cpp", ".cu")):
    for root in roots:
        yield from (p for p in sorted((REPO_ROOT / root).rglob("*")) if p.is_file() and p.suffix in suffixes)

def function(text, name):
    """Extract a named native definition, including nested scopes."""
    match = re.search(r"\b" + re.escape(name) + r"\s*\([^;{}]*\)\s*(?:const\s*)?\{", text)
    assert match, name
    start = match.end()
    depth = 1
    for end in range(start, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return text[start:end]
    raise AssertionError("unclosed function " + name)

def cmake_list(name):
    text = re.sub(r"(?m)#[^\n]*", "", read_text("CMakeLists.txt"))
    match = re.search(r"set\(\s*" + re.escape(name) + r"\s(.*?)\n\s*\)", text, re.S)
    assert match, "CMakeLists.txt has no set(%s ...) block" % name
    return match.group(1).split()
