"""Pure FFprobe metadata conversion tests without invoking the executable."""

import pytest

from shared.exceptions.ai import FFprobeUnavailableError
from shared.rendering.ffprobe import _frame_rate


def test_fractional_frame_rate_is_parsed() -> None:
    assert _frame_rate("30000/1001") == pytest.approx(29.970, rel=0.001)


def test_invalid_frame_rate_is_rejected() -> None:
    with pytest.raises(FFprobeUnavailableError):
        _frame_rate("invalid")
