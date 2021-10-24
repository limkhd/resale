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
