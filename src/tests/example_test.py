import pytest

# Example test


@pytest.mark.parametrize(
    "arg, answer",
    [
        (3.1, 3),
        (5, 5),
        (2.6, 2),
    ],
)
def test_int(arg, answer):
    assert int(arg) == answer
