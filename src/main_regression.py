import logging
import os
from copy import deepcopy
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import lightgbm as lgb
import sklearn.model_selection as ms
from sklearn.preprocessing import OrdinalEncoder
import sklearn.metrics as m

logger = logging.getLogger(__name__)


def get_best_params(tuningparams, regressor, X, y, n_splits=5):
    # Perform hyperparameter optimization on one parameter at at time, like coordinate descent
    best_params = {}
    for p in tuningparams:
        cv_iter = ms.KFold(n_splits=n_splits, shuffle=True).split(X, y)
        tuningparams_ = {p: tuningparams[p]}
        cv_obj = ms.GridSearchCV(
            regressor,
            tuningparams_,
            cv=cv_iter,
            scoring="neg_root_mean_squared_error",
            verbose=1,
        )
        cv_obj.fit(X, y)
        best_params.update(deepcopy(cv_obj.best_params_))

    return best_params


def do_feature_engineering(df, categorical_columns, train=True, ordinal_encs=None):

    df_out = df.copy()

    if train:
        ordinal_encs_ = {}
    else:
        ordinal_encs_ = ordinal_encs

    for c in categorical_columns:
        if train:
            enc = OrdinalEncoder()
            df_out[c] = enc.fit_transform(df_out[c].values.reshape(-1, 1)).astype(int)
            ordinal_encs_[c] = enc
            df_out[c] = df_out[c].astype("category")
        else:
            ordinal_encs_ = ordinal_encs
            df_out[c] = (
                ordinal_encs_[c].transform(df_out[c].values.reshape(-1, 1)).astype(int)
            )
            df_out[c] = df_out[c].astype("category")

    cols_to_drop = [
        "street_name",
        "block",
        "month",
        "storey_range",
        "lease_commence_date",
        "rem_lease_at_sale",
        "quarter",
        "address",
        "POSTAL",
        "LATITUDE",
        "LONGITUDE",
        "nearest_station",
        "nearest_station_lat",
        "nearest_station_lng",
        "flat_model",
        "sale_year",
    ]
    df_out = df_out.drop(columns=cols_to_drop)
    return df_out, ordinal_encs_


def main():
    # use_cv = True
    use_cv = False

    in_file = "../data/processed/resales_with_latlong.csv"

    if not os.path.exists(in_file):
        print("%s does not exist, please run generate_data.py to generate it" % in_file)

    df_raw = pd.read_csv(in_file)
    logger.info("Total resale transactions: %d" % len(df_raw))

    numpy.random.seed(0)

    df = df_raw.sort_values(["month", "town", "street_name"])

    df = df[:-50000]
    logger.info("Most recent transactions:")
    print(df.tail(10))

    df_test = df.iloc[-50000:]
    df = df.iloc[:-50000]

    X = df.drop(columns="resale_price")
    y = df["resale_price"].values
    Xtest = df_test.drop(columns="resale_price")
    ytest = df_test["resale_price"].values

    categorical_columns = ["town", "flat_type", "flat_model"]
    X, encoders = do_feature_engineering(X, categorical_columns, train=True)
    Xtest, _ = do_feature_engineering(
        Xtest, categorical_columns, train=False, ordinal_encs=encoders
    )

    logger.info("Train head/tail:")
    print(X.head(3))
    print(X.tail(3))
    # df = df.iloc[:-20000]
    # df = df_raw.sample(100000)
    logger.info("Test head/tail:")
    print(Xtest.head(3))
    print(Xtest.tail(3))

    X = X.iloc[-50000:]
    y = y[-50000:]

    tuningparams = {
        "learning_rate": [1, 0.5, 0.1, 0.05, 0.01, 0.001],
        "n_estimators": [50, 100, 200, 400, 800],
        "num_leaves": [4, 8, 16],
        "max_depth": [8, 16],
        "colsample_bytree": [0.8, 1],
        "subsample": [0.05, 0.1, 0.25, 0.5, 0.75, 1],
        "reg_alpha": [0.00001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10],
        "reg_lambda": [0.0001, 0.001, 0.01, 0.1, 1, 10],
    }

    tuningparams = {
        "learning_rate": [0.5],
        "n_estimators": [3200],
        "num_leaves": [8],
        "max_depth": [16],
        "colsample_bytree": [0.8, 1],
        "subsample": [0.5, 0.25],
        "min_data_in_leaf": [100],
        "reg_alpha": [0.00001],
        "reg_lambda": [0.01],
    }

    best_params_default = {
        "learning_rate": 0.5,
        "n_estimators": 3200,
        "num_leaves": 8,
        "max_depth": 16,
        "colsample_bytree": 0.8,
        "subsample": 0.5,
        "min_data_in_leaf": 100,
        "reg_alpha": 0.00001,
        "reg_lambda": 0.01,
    }

    regressor = lgb.LGBMRegressor()

    # Search params
    if use_cv:
        best_params = get_best_params(tuningparams, regressor, X, y, n_splits=2)
        pd.to_pickle(best_params, "best_params.pickle")
    else:
        if os.path.exists("best_params.pickle"):
            best_params = pd.read_pickle("best_params.pickle")
        else:
            best_params = best_params_default

    print(best_params)
    best_regressor = lgb.LGBMRegressor(**best_params)
    best_regressor.fit(X, y, eval_set=[(X, y), (Xtest, ytest)], verbose=800)
    # lgb.plot_importance(best_regressor)
    # lgb.plot_metric(best_regressor)
    # lgb.plot_tree(best_regressor, tree_index=1)
    plt.show()

    ypred = best_regressor.predict(Xtest)
    ypred_train = best_regressor.predict(X)

    ypred_res = ypred - ytest

    # Get flat indices with highest residuals
    # limit = 500
    flats_sorted_by_residual = numpy.argsort(ypred_res)

    worst_predictions = {}
    worst_predictions["undervalued"] = flats_sorted_by_residual[::-1]
    worst_predictions["overvalued"] = flats_sorted_by_residual

    for key in ["undervalued", "overvalued"]:
        Xtest_worst = Xtest.iloc[worst_predictions[key]]
        ypred_worst = ypred[worst_predictions[key]]
        ytest_worst = ytest[worst_predictions[key]]

        df_worst = df_raw.iloc[Xtest_worst.index].copy()

        df_worst["predicted"] = ypred_worst
        df_worst["predicted"] = df_worst["predicted"].astype(int)

        core_columns = [
            "month",
            "town",
            "flat_type",
            "block",
            "street_name",
            "storey_range",
            "rem_lease",
            "nearest_station",
            "distance_to_mrt",
        ]
        pred_columns = ["resale_price", "predicted"]

        df_worst_v2 = df_worst[core_columns + pred_columns]
        # print((ypred_worst - ytest_worst)[:10])

    print(
        "Train MAPE is %.3f%%"
        % (m.mean_absolute_percentage_error(y, ypred_train) * 100)
    )
    print(
        "Test MAPE is %.3f%%" % (m.mean_absolute_percentage_error(ytest, ypred) * 100)
    )


if __name__ == "__main__":
    main()
