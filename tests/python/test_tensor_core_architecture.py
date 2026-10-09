"""Tensor ownership diagnostics and representation-safety regression."""

from tests.support.source import read_cpp_source as _source
from tests.support.source import REPO_ROOT
from pathlib import Path
import re

import orthon
import pytest


REPO = REPO_ROOT




def test_tensor_impl_has_only_the_four_canonical_component_owners():
    # Inspect member types, not member names, accessor formatting, or ordering.
    source = _source("include/orthon/core/tensor_impl.h")
    members = source.split("private:", 1)[1].split("};", 1)[0]
    declarations = [part.strip() for part in members.split(";") if part.strip()]
    types = [re.sub(r"\s+[A-Za-z_]\w*$", "", part) for part in declarations]
    assert sorted(types) == sorted([
        "std::shared_ptr<Buffer>", "Layout", "TensorMeta", "AutogradMeta",
    ])


def test_storage_is_an_exclusive_allocation_bridge_without_dtype_or_generation():
    buffer = _source("include/orthon/core/buffer.h")
    storage = _source("include/orthon/core/storage.h")
    saved_versions = _source("include/orthon/autograd/node.h")
    assert re.search(r"\bAllocation\s+\w+\s*;", buffer)
    assert re.search(r"\bAllocation\s+\w+\s*;", storage)
    assert "Storage" not in buffer
    assert not re.search(r"\b(?:DType|TensorMeta)\b", buffer + storage)
    assert not re.search(r"\b(?:atomic|version|bump_version)\b", storage)
    saved = saved_versions.split("struct SavedTensorVersion", 1)[1].split("};", 1)[0]
    assert re.search(r"std::shared_ptr\s*<\s*Buffer\s*>", saved)
    assert "Storage" not in saved


def test_flatten_checks_shape_product_without_a_large_allocation():
    # This valid empty representation allocates zero bytes. Flattening just
    # the two nonempty axes would overflow even though total numel is zero.
    limit = (1 << 63) - 1
    empty = orthon.zeros([limit, 2, 0])
    assert empty.numel == 0
    with pytest.raises(OverflowError, match="product overflows"):
        empty.flatten(0, 1)
    assert list(empty.flatten().shape) == [0]
