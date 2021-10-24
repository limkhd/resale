import math
import pytest
from de_utils import expand_address_with_abbreviations


@pytest.mark.parametrize(
    "address, answer",
    [
        (
            "NORTH BUONA VISTA RD",
            [
                "NTH BUONA VISTA RD",
                "NTH BUONA VISTA ROAD",
                "NORTH BUONA VISTA RD",
                "NORTH BUONA VISTA ROAD",
            ],
        ),
        (
            "NTH BUONA VISTA RD",
            [
                "NTH BUONA VISTA RD",
                "NTH BUONA VISTA ROAD",
                "NORTH BUONA VISTA RD",
                "NORTH BUONA VISTA ROAD",
            ],
        ),
        (
            "UPP BT TIMAH RD",
            [
                "UPP BT TIMAH RD",
                "UPPER BT TIMAH RD",
                "UPP BUKIT TIMAH RD",
                "UPPER BUKIT TIMAH RD",
                "UPP BT TIMAH ROAD",
                "UPPER BT TIMAH ROAD",
                "UPP BUKIT TIMAH ROAD",
                "UPPER BUKIT TIMAH ROAD",
            ],
        ),
    ],
)
def test_abbreviation(address, answer):
    assert sorted(expand_address_with_abbreviations(address)) == sorted(answer)
