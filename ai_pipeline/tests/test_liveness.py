import numpy as np
import pytest

from liveness.active_liveness import ActiveLiveness
from liveness.passive_liveness import PassiveLiveness


def test_passive_liveness_not_implemented(sample_rgb_array):
    checker = PassiveLiveness()
    with pytest.raises(NotImplementedError):
        checker.score(sample_rgb_array)


def test_active_liveness_not_implemented(sample_rgb_array):
    checker = ActiveLiveness()
    with pytest.raises(NotImplementedError):
        checker.verify_challenge([sample_rgb_array], "blink")
