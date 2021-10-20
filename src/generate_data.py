import datetime
import os
import pandas as pd
import zipfile
import requests
import json
from tqdm import tqdm
from itertools import product
from database import SQL3DB
from ratelimit import limits, RateLimitException, sleep_and_retry

from abbreviations import abbrev_expansion_dict


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


def add_engineered_features(df, rpi_df):

    df["sale_year"] = df["month"].map(lambda x: int(x.split("-")[0]))
    df["rem_lease"] = df["lease_commence_date"] + 99 - datetime.datetime.now().year
    df["rem_lease_at_sale"] = (df["lease_commence_date"] + 99 - df["sale_year"]).map(
        lambda x: min(x, 98)
    )
    df["mid_band_storey"] = df["storey_range"].map(storey_range_to_numeric)
    df["quarter"] = df["month"].map(month_to_quarter)
    df = df.merge(rpi_df, on="quarter")
    df = df.rename(columns={"index": "RPI"})

    return df


def merge_csv_records(data_dir):

    filenames = [
        # "resale-flat-prices-based-on-approval-date-1990-1999.csv",
        "resale-flat-prices-based-on-approval-date-2000-feb-2012.csv",
        "resale-flat-prices-based-on-registration-date-from-mar-2012-to-dec-2014.csv",
        "resale-flat-prices-based-on-registration-date-from-jan-2015-to-dec-2016.csv",
        "resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv",
    ]

    df_all = None
    for f in filenames:
        df = pd.read_csv("%s/%s" % (data_dir, f))
        if df_all is None:
            df_all = df
        else:
            df_all = pd.concat((df_all, df), axis=0)

    df_all["address"] = df_all["block"] + " " + df_all["street_name"]
    df_all = df_all.drop(columns=["remaining_lease"])
    return df_all


def expand_address_with_abbreviations(address, abbrev_expansion_dict):
    try:
        input_tokens = address.split(" ")
    except ValueError:
        print("Error tokenizing %s" % address)
        return ""

    output_tokens = []
    for t in input_tokens:
        if t in abbrev_expansion_dict:
            output_tokens.append(abbrev_expansion_dict[t])
        else:
            output_tokens.append([t])

    expanded_addresses = [" ".join(p) for p in product(*output_tokens)]
    return expanded_addresses


def verify_onemap_addresses(addr_latlong_df):
    """Returns a dataframe of addresses where the query did not match OneMap result.
    """

    found_addresses = addr_latlong_df["BLK_NO"] + " " + addr_latlong_df["ROAD_NAME"]
    query_addresses = addr_latlong_df["QUERY_ADDRESS"]

    errors = []
    for found_addr, query_addr in tqdm(zip(found_addresses, query_addresses)):

        # Get expanded list of query addresses if there are common abbreviations
        # This is due to inconsistence in abbreviations being used for the same streets
        # from HDB and OneMap
        expanded_query_addresses = expand_address_with_abbreviations(
            query_addr, abbrev_expansion_dict
        )
        if found_addr not in expanded_query_addresses:
            errors.append({"found": found_addr, "query": expanded_query_addresses[0]})

    errors_df = pd.DataFrame.from_records(errors)
    return errors_df


def download_url(url, save_path, chunk_size=1024):
    """Downloads url to save_path.
    """

    r = requests.get(url, stream=True)
    total_size_in_bytes = int(r.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(save_path, "wb") as fd:
        for chunk in tqdm(r.iter_content(chunk_size=chunk_size)):
            progress_bar.update(len(chunk))
            fd.write(chunk)


@sleep_and_retry
@limits(calls=200, period=60)
def call_onemap_api(query_address):
    """Calls OneMap API to get lat/long from query address string. Decorated with
    rate limiter.
    """
    query_string = (
        "https://developers.onemap.sg/commonapi/search?searchVal="
        + str(query_address)
        + "&returnGeom=Y&getAddrDetails=Y"
    )
    resp = requests.get(query_string).content
    return resp


def get_addr_latlong_df(addresses, db, missing_data_path):
    missing = []

    # If missing_data_path exists, load list of missing addresses
    if os.path.exists(missing_data_path):
        with open(missing_data_path, "r") as f:
            for line in f:
                missing.append(line.strip())

    # print("Processing unique addresses and querying OneMap for latitude/longitude")
    for query_address in tqdm(addresses):
        if db.key_in_db((query_address,)) or query_address in missing:
            continue
        else:

            # Assume query is missing
            query_is_missing = True

            # Call API
            resp = call_onemap_api(query_address)
            json_resp = json.loads(resp)

            # Got non-empty results
            if json_resp["found"] >= 1:
                query_block = query_address.split(" ")[0]
                assert query_block[0].isnumeric()

                for i in range(len(json_resp["results"])):
                    address_record = json_resp["results"][i]

                    # Found exact block match
                    if address_record["BLK_NO"] == query_block:
                        query_is_missing = False

            # Add to missing if not found, else store in DB
            if query_is_missing:
                missing.append(query_address)
            else:
                address_record["QUERY_ADDRESS"] = query_address
                db.store_dict_in_db(address_record)

    addr_latlong_df = db.read_df_from_db(db.db_path)
    print("Saving to %s" % missing_data_path)
    with open(missing_data_path, "w") as f:
        for addr in missing:
            print(addr, file=f)

    return addr_latlong_df


def get_RPI_data(data_dir, overwrite_data=False):
    if not os.path.exists(data_dir):
        print("Creating %s" % data_dir)
        os.makedirs(data_dir)

    main_data_path = (
        "https://storage.data.gov.sg/hdb-resale-price-index/hdb-resale-price-index.zip"
    )
    local_zip_path = "%s/hdb-resale-price-index.zip" % data_dir
    if not os.path.exists(local_zip_path) or overwrite_data is True:
        print("Downloading file")
        download_url(main_data_path, local_zip_path)
        # wget.download(main_data_path, data_dir)

    csv_filename = (
        "housing-and-development-board-resale-price-index-1q2009-100-quarterly.csv"
    )
    if not os.path.exists("%s/%s" % (data_dir, csv_filename)) or overwrite_data is True:
        print("Unzipping files")
        with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        print("Done")


def get_resale_transaction_data(data_dir, overwrite_data=False):
    if not os.path.exists(data_dir):
        print("Creating %s" % data_dir)
        os.makedirs(data_dir)

    main_data_path = (
        "https://storage.data.gov.sg/resale-flat-prices/resale-flat-prices.zip"
    )
    local_zip_path = "%s/resale-flat-prices.zip" % data_dir
    if not os.path.exists(local_zip_path) or overwrite_data is True:
        print("Downloading file")
        download_url(main_data_path, local_zip_path)

    csv_filename = "resale-flat-prices-based-on-approval-date-2000-feb-2012.csv"
    if not os.path.exists("%s/%s" % (data_dir, csv_filename)) or overwrite_data is True:
        print("Unzipping files")
        with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        print("Done")


def storey_range_to_numeric(storey_range):
    low, high = storey_range.split(" TO ")
    return float((int(low) + int(high)) / 2)


def main():
    data_dir = "../data"
    db_path = "%s/postal_codes.db" % data_dir
    missing_onemap_data_path = "%s/missing.txt" % data_dir

    # Initialize DB with OneMap schema
    schema = {
        "columns": {
            "SEARCHVAL": "TEXT",
            "BLK_NO": "TEXT",
            "ROAD_NAME": "TEXT",
            "BUILDING": "TEXT",
            "ADDRESS": "TEXT",
            "POSTAL": "TEXT",
            "X": "TEXT",
            "Y": "TEXT",
            "LATITUDE": "TEXT",
            "LONGITUDE": "TEXT",
            "LONGTITUDE": "TEXT",
            "QUERY_ADDRESS": "TEXT",
        },
        "primary_key": ["QUERY_ADDRESS"],
    }
    db = SQL3DB(db_path, schema)

    # Download and extract latest data from data.gov.sg
    overwrite_data = False
    get_resale_transaction_data(data_dir, overwrite_data=overwrite_data)
    get_RPI_data(data_dir, overwrite_data=overwrite_data)
    rpi_filename = (
        "housing-and-development-board-resale-price-index-1q2009-100-quarterly.csv"
    )
    rpi_df = pd.read_csv("%s/%s" % (data_dir, rpi_filename))

    # Get combined dataframe from multiple raw files
    df_all = merge_csv_records(data_dir)

    # Get lat/long/postal from OneMap
    unique_addresses = get_unique_addresses_from_df(df_all)
    addr_latlong_df = get_addr_latlong_df(
        unique_addresses, db, missing_onemap_data_path
    )

    # Verify queried and retrieved addresses are the same to ensure lat/long are
    # correct
    print("Verifying consistency of queried and retrieved addresses")
    errors_df = verify_onemap_addresses(addr_latlong_df)

    # Merge resale dataset with postal codes and lat/long, while removing those
    # with invalid postal codes
    addr_latlong_df_trimmed = addr_latlong_df[
        ["POSTAL", "LATITUDE", "LONGITUDE", "QUERY_ADDRESS"]
    ]
    df_with_latlong = df_all.merge(
        addr_latlong_df_trimmed, left_on="address", right_on="QUERY_ADDRESS", how="left"
    )
    df_with_latlong = df_with_latlong.dropna(subset=["POSTAL"]).drop(
        columns="QUERY_ADDRESS"
    )

    # Remove problematic addresses (those in errors_df)
    df_with_latlong = df_with_latlong[
        ~df_with_latlong["address"].isin(errors_df["query"])
    ]

    # Add engineered features
    df_with_latlong = add_engineered_features(df_with_latlong, rpi_df)

    # Write augmented dataframes
    out_file = "%s/processed/resales_with_latlong.csv" % data_dir
    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))
    print("Saving to %s" % out_file)
    df_with_latlong.to_csv(out_file, index=False)


if __name__ == "__main__":
    main()
