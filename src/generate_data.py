import datetime
import logging
import os
import sys
import numpy as np
import pandas as pd
import yaml
import zipfile
import requests
import json
from tqdm import tqdm
from resale.database import SQL3DB
from resale.de_utils import (
    month_to_quarter,
    storey_range_to_numeric,
    get_unique_addresses_from_df,
    expand_address_with_abbreviations,
)
from ratelimit import limits, sleep_and_retry
from scipy.spatial.distance import cdist

format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=format)
root_logger = logging.getLogger()

logger = logging.getLogger(__name__)


def get_depreciation_tables(r):

    # freehold value
    fv = 1.0 / r

    years = np.arange(99)
    lv = (1 - (1 + r) ** -years) / r
    tenure_disc_values = lv[::-1] / fv
    remaining_lease_disc_values = lv / fv

    return tenure_disc_values, remaining_lease_disc_values


def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees)

    All args must be of equal length.

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def add_MRT_distance(df_with_latlong, mrt_data_df):
    """Adds distance to nearest MRT to each resale flat"""

    # Convert mrt_points and data_points to 2-d tuples
    mrt_points = [(x, y) for x, y in zip(mrt_data_df["lat"], mrt_data_df["lng"])]
    data_points = [
        (x, y)
        for x, y in zip(df_with_latlong["LATITUDE"], df_with_latlong["LONGITUDE"])
    ]

    # Get nearest MRT station for each using Euclidean distance of lat/lng coordinates
    # Approximate is reasonable since SG is so small and on the equator so ordering should not be affected
    nearest_mrt_stations = cdist(data_points, mrt_points).argmin(axis=1)

    # Copy information into df_with_latlong
    nearest_mrt_data = mrt_data_df.iloc[nearest_mrt_stations].reset_index(drop=True)
    df_with_latlong["nearest_station"] = nearest_mrt_data["station_name"].values
    df_with_latlong["nearest_station_lat"] = nearest_mrt_data["lat"].values
    df_with_latlong["nearest_station_lng"] = nearest_mrt_data["lng"].values

    assert df_with_latlong.isnull().any(axis=1).sum() == 0

    # Once we find the closest station, we recalculate the actual haversine distance
    df_with_latlong["distance_to_mrt"] = haversine_np(
        df_with_latlong["LONGITUDE"],
        df_with_latlong["LATITUDE"],
        df_with_latlong["nearest_station_lng"],
        df_with_latlong["nearest_station_lat"],
    )

    return df_with_latlong


def add_engineered_features(df):

    df["sale_year"] = df["month"].map(lambda x: int(x.split("-")[0]))
    df["rem_lease"] = df["lease_commence_date"] + 99 - datetime.datetime.now().year
    df["rem_lease_at_sale"] = (df["lease_commence_date"] + 99 - df["sale_year"]).map(
        lambda x: min(x, 98)
    )
    df["mid_band_storey"] = df["storey_range"].map(storey_range_to_numeric)
    df["quarter"] = df["month"].map(month_to_quarter)

    return df


def merge_csv_records(main_data_dir):

    filenames = [
        # "resale-flat-prices-based-on-approval-date-1990-1999.csv",
        "resale-flat-prices-based-on-approval-date-2000-feb-2012.csv",
        "resale-flat-prices-based-on-registration-date-from-mar-2012-to-dec-2014.csv",
        "resale-flat-prices-based-on-registration-date-from-jan-2015-to-dec-2016.csv",
        "resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv",
    ]

    df_all = None
    for f in filenames:
        df = pd.read_csv("%s/interim/%s" % (main_data_dir, f))
        if df_all is None:
            df_all = df
        else:
            df_all = pd.concat((df_all, df), axis=0)

    df_all["address"] = df_all["block"] + " " + df_all["street_name"]
    df_all = df_all.drop(columns=["remaining_lease"])
    return df_all


def verify_onemap_addresses(addr_latlong_df):
    """Returns a dataframe of addresses where the query did not match OneMap result."""

    found_addresses = addr_latlong_df["BLK_NO"] + " " + addr_latlong_df["ROAD_NAME"]
    query_addresses = addr_latlong_df["QUERY_ADDRESS"]

    errors = []
    for found_addr, query_addr in tqdm(zip(found_addresses, query_addresses)):

        # Get expanded list of query addresses if there are common abbreviations
        # This is due to inconsistence in abbreviations being used for the same streets
        # from HDB and OneMap
        expanded_query_addresses = expand_address_with_abbreviations(query_addr)
        if found_addr not in expanded_query_addresses:
            errors.append({"found": found_addr, "query": expanded_query_addresses[0]})

    errors_df = pd.DataFrame.from_records(errors)
    return errors_df


def download_url(url, save_path, chunk_size=1024):
    """Downloads url to save_path."""

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
    """Returns dataframe containing addresses, their postal code and longitude obtained from OneMap API

    Parameters
    ----------
    addresses : List-like
        List of addresses to query OneMap API
    db : SQL3DB
        SQL3DB instance from database.py to persist obtained data
    missing_data_path : TODO
        Path to text file to store addresses that cannot be located in OneMap (too old, demolished etc)

    Returns
    -------
    pandas.DataFrame
        DataFrame with query and response columns

    """
    missing = []

    # If missing_data_path exists, load list of missing addresses
    if os.path.exists(missing_data_path):
        with open(missing_data_path, "r") as f:
            for line in f:
                missing.append(line.strip())

    # logger.info("Processing unique addresses and querying OneMap for latitude/longitude")
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

    # Convert latitude/longitude to float
    addr_latlong_df["LATITUDE"] = addr_latlong_df["LATITUDE"].astype(float)
    addr_latlong_df["LONGITUDE"] = addr_latlong_df["LONGITUDE"].astype(float)

    logger.info("Saving to %s" % missing_data_path)
    with open(missing_data_path, "w") as f:
        for addr in missing:
            print(addr, file=f)

    return addr_latlong_df


def get_resale_transaction_data(main_data_dir, overwrite_data=False):
    if not os.path.exists(main_data_dir):
        logger.info("Creating %s" % main_data_dir)
        os.makedirs(main_data_dir)

    main_data_path = (
        "https://storage.data.gov.sg/resale-flat-prices/resale-flat-prices.zip"
    )
    local_zip_path = "%s/raw/resale-flat-prices.zip" % main_data_dir
    if not os.path.exists(local_zip_path) or overwrite_data is True:
        logger.info("Downloading file")
        download_url(main_data_path, local_zip_path)

    csv_filename = "resale-flat-prices-based-on-approval-date-2000-feb-2012.csv"
    if (
        not os.path.exists("%s/%s" % (main_data_dir, csv_filename))
        or overwrite_data is True
    ):
        logger.info("Unzipping files")
        with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
            zip_ref.extractall(main_data_dir + "/interim")
        logger.info("Done")


def main():
    parameters_file = sys.argv[1]

    with open(parameters_file, "r") as f:
        params = yaml.safe_load(f)

    de_options = params["data_engineering_options"]
    main_data_dir = de_options["main_data_dir"]

    # Download and extract latest data from data.gov.sg
    logger.info("Downloading latest data from data.gov.sg")
    overwrite_data = False
    get_resale_transaction_data(main_data_dir, overwrite_data=overwrite_data)

    # Get combined dataframe from multiple raw files
    logger.info("Merging all files into one")
    df_all = merge_csv_records(main_data_dir)

    logger.info("Total transactions is %d" % len(df_all))

    # Get lat/long/postal from OneMap
    logger.info("Getting postal codes and lat/long from OneMap API")
    db_path = "%s/%s" % (main_data_dir, de_options["db_name"])
    missing_onemap_data_path = "%s/%s" % (
        main_data_dir,
        de_options["missing_onemap_data_filename"],
    )
    db = SQL3DB(db_path, de_options["onemap_db_schema"])
    unique_addresses = get_unique_addresses_from_df(df_all)
    addr_latlong_df = get_addr_latlong_df(
        unique_addresses, db, missing_onemap_data_path
    )

    # Merge resale dataset with postal codes and lat/long, while removing those
    # with invalid postal codes
    df_with_latlong = df_all.merge(
        addr_latlong_df[["POSTAL", "LATITUDE", "LONGITUDE", "QUERY_ADDRESS"]],
        left_on="address",
        right_on="QUERY_ADDRESS",
        how="left",
    ).drop(columns="QUERY_ADDRESS")
    df_with_latlong = df_with_latlong.dropna(subset=["POSTAL"])

    # Verify queried and retrieved addresses are the same to ensure lat/long are
    # correct
    logger.info("Verifying consistency of queried and retrieved addresses")
    errors_df = verify_onemap_addresses(addr_latlong_df)

    logger.info("Rows with errors (likely demolished):\n")
    print(errors_df)
    # Remove problematic addresses (those in errors_df) - typically demolished
    df_with_latlong = df_with_latlong[
        ~df_with_latlong["address"].isin(errors_df["query"])
    ]

    # Add engineered features
    logger.info("Adding engineered features")
    df_with_latlong = add_engineered_features(df_with_latlong)

    # align with Bala's table
    logger.info("Adding lease depreciation factors")
    r = de_options["bala_discount_factor"]
    tenure_disc_values, remaining_lease_disc_values = get_depreciation_tables(r)
    df_with_latlong["lease_depreciation_factor"] = df_with_latlong["rem_lease"].map(
        lambda x: remaining_lease_disc_values[x]
    )
    df_with_latlong["lease_depreciation_factor_at_sale"] = df_with_latlong[
        "rem_lease_at_sale"
    ].map(lambda x: remaining_lease_disc_values[x])

    # Add distance to nearest MRT
    logger.info("Adding distance to nearest MRT")
    mrt_data_path = "%s/%s" % (main_data_dir, de_options["mrt_data_filename"])
    mrt_data_df = pd.read_csv(mrt_data_path)
    mrt_data_df = mrt_data_df[mrt_data_df["type"] == "MRT"]
    df_with_latlong = add_MRT_distance(df_with_latlong, mrt_data_df)

    # Write augmented dataframes
    out_file = "%s/processed/resales_with_latlong.csv" % main_data_dir
    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))
    logger.info("Saving to %s" % out_file)
    df_with_latlong.to_csv(out_file, index=False)


if __name__ == "__main__":
    main()
