import pytest


@pytest.mark.parametrize("dtype", ["float32", "float16"], ids=["cpu-fp32-training", "cpu-fp16-storage-transfer"])
def test_local_amp_consensus_skips_every_rank_then_recovers_master_and_scaler_progress(distributed_job, dtype):
    results = distributed_job("amp", dtype=dtype)
    for rank in range(2):
        for iteration, step in enumerate(results[rank]):
            assert step["scale"] == [8., 2., 2., 8.][iteration]
            assert step["steps"] == [1, 1, 2, 3][iteration]
            assert step["found_inf"] is (iteration == 1)
            assert step["scaler_after_update"]["found_inf"] is False
            assert step["scaler_after_update"] == results[1-rank][iteration]["scaler_after_update"]
            assert step["masters"] == results[1-rank][iteration]["masters"]
            assert step["moments"] == results[1-rank][iteration]["moments"]
            if iteration == 1:
                assert step["masters"] == step["before"]
                assert step["model"] == results[rank][0]["model"]
            else:
                assert step["grads"][0]["values"] == [4., 8., 12., 20.]
                for master, gradient in zip(step["masters"], step["grads"]):
                    assert master["dtype"] == gradient["dtype"] == "float32"
                    assert master["is_leaf"] and master["requires_grad"]
                    assert gradient["requires_grad"] is False
