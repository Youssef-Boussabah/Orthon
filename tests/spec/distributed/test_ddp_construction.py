import pytest


@pytest.mark.parametrize("source", [0, 1])
def test_ddp_broadcasts_named_parameters_but_preserves_local_buffers_and_module_delegation(
    distributed_job, source,
):
    results = distributed_job("facade", src=source)
    for rank, result in enumerate(results):
        for name, initial in {"a": [1., 2., 3., 4.], "b": [1., 2.], "c": [2.]}.items():
            entry = result["initial"][name]
            assert entry["values"] == [v + 10 * source for v in initial]
            assert entry["dtype"] == "float32" and entry["requires_grad"] and entry["is_leaf"]
        assert result["initial"]["offset"]["values"] == [10. + rank]
        assert result["parameter_count"] == 3
        assert all(entry["values"] == [.5] * len(entry["values"]) for entry in result["loaded"].values())
    assert results[1]["output"] - results[0]["output"] == 1
