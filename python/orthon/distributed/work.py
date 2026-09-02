class Work:
    def wait(self):
        raise NotImplementedError

    def is_completed(self):
        raise NotImplementedError


class _CompletedWork(Work):
    def wait(self):
        return True

    def is_completed(self):
        return True


__all__ = ["Work"]
