from abbreviations import abbrev_expansion_dict
import logging
from itertools import product

logger = logging.getLogger(__name__)


def storey_range_to_numeric(storey_range):
    low, high = storey_range.split(" TO ")
    return float((int(low) + int(high)) / 2)


def month_to_quarter(x):
    year, month = x.split("-")
    mtoq = {
        "01": "Q1",
        "02": "Q1",
        "03": "Q1",
        "04": "Q2",
        "05": "Q2",
        "06": "Q2",
        "07": "Q3",
        "08": "Q3",
        "09": "Q3",
        "10": "Q4",
        "11": "Q4",
        "12": "Q4",
    }
    return "-".join((year, mtoq[month]))


def get_unique_addresses_from_df(df_all):
    addresses = (
        df_all[["block", "street_name"]]
        .drop_duplicates()
        .sort_values(["street_name", "block"])
    )
    addresses = addresses["block"] + " " + addresses["street_name"]
    return addresses


def expand_address_with_abbreviations(address):
    """Generate all possible non-abbreviated or abbreviated forms of an address

    Example: "NTH BUONA VISTA RD" ->
    ["NTH BUONA VISTA RD", "NTH BUONA VISTA ROAD",
    "NORTH BUONA VISTA RD", "NORTH BUONA VISTA ROAD"]

    This is done by establishing equivalence sets for each abbreviation, then
    using itertools.product to generate all possibilities
    """

    try:
        input_tokens = address.split(" ")
    except ValueError:
        logger.info("Error tokenizing %s" % address)
        return ""

    output_tokens = []
    for t in input_tokens:
        output_tokens.append(
            abbrev_expansion_dict[t] if t in abbrev_expansion_dict else [t]
        )

    expanded_addresses = [" ".join(p) for p in product(*output_tokens)]
    return expanded_addresses
