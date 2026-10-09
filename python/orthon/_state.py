"""Private state-owner preparation protocol, independent of file serialization."""


class _PreparedLoad:
    def __init__(self, commit, rollback, placement=None):
        self.commit = commit
        self.rollback = rollback
        self.placement = placement


def _commit_prepared(participants):
    """Publish prepared owners in order, undo synchronous failures in reverse."""
    attempted = []
    try:
        for participant in participants:
            # Include a participant whose commit may fail after a partial write.
            attempted.append(participant)
            participant.commit()
    except BaseException as original:
        failures = []
        for participant in reversed(attempted):
            try:
                participant.rollback()
            except BaseException as error:
                failures.append(error)
        if failures:
            raise RuntimeError(
                "checkpoint: transaction rollback failed (%s)" %
                "; ".join(str(error) for error in failures)
            ) from original
        raise
