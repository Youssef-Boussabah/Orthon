"""Shared collective training profiles; parameter IDs identify stable fixtures."""
import numpy as np
import pytest


MODES = [pytest.param({"manual": True}, id="manual"),
         pytest.param({}, id="automatic"),
         pytest.param({"bucket": .000001}, id="bucket-small"),
         pytest.param({"bucket": 1.}, id="bucket-large"),
         pytest.param({"bucket": .000001, "overlap": True}, id="overlap-small"),
         pytest.param({"bucket": 1., "overlap": True}, id="overlap-large")]
COEFFICIENTS = {"a": np.array([1., 2., 3., 5.]), "b": np.array([2., -1.]), "c": np.array([.5])}
INITIAL = {"a": np.array([1., 2., 3., 4.]), "b": np.array([1., 2.]), "c": np.array([2.])}
