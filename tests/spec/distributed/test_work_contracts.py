def test_local_async_work_is_complete_on_return_and_repeated_wait_does_not_modify_values(distributed_job):
    results = distributed_job("collectives")
    for blocking, asynchronous in results:
        assert blocking["work"] is None
        assert asynchronous["work"] == {"initial": True, "waits": [True, True], "done": True}
        assert blocking["summed"] == asynchronous["summed"]
        assert blocking["broadcasts"] == asynchronous["broadcasts"]
