import sys
import logging
import os
from copy import deepcopy
from resale.preprocessor import Preprocessor
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import lightgbm as lgb
import sklearn.model_selection as ms
import sklearn.metrics as m
from resale.utils import read_yml

log_format = (
    "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
)
date_format = "%H:%M:%S"
logging.basicConfig(
    stream=sys.stdout, level=logging.INFO, format=log_format, datefmt=date_format
)

logger = logging.getLogger(__name__)


def get_best_params(tuningparams, regressor, X, y, n_splits=3):
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


def main():
    params = read_yml(sys.argv[1])

    modeling_options = params["modeling_options"]
    preprocessing_options = params["preprocessing_options"]
    split_options = params["split_options"]

    numpy.random.seed(0)

    in_file = "../data/processed/resales_with_latlong.csv"

    if not os.path.exists(in_file):
        print("%s does not exist, please run generate_data.py to generate it" % in_file)

    df_raw = pd.read_csv(in_file)
    logger.info("Total resale transactions: %d" % len(df_raw))

    df = df_raw.sort_values(["month", "town", "street_name"]).copy()

    test_size = split_options["test_size"]
    valid_size = split_options["valid_size"]

    # Create test split
    df_test = df.iloc[-test_size:]
    df = df.iloc[:-test_size]

    # Create validation split
    df_val = df.iloc[-valid_size:]
    df = df.iloc[:-valid_size]

    p = Preprocessor(**preprocessing_options)
    X, y = p.generate_model_inputs(df, train=True)
    Xtest, ytest = p.generate_model_inputs(df_test)
    Xval, yval = p.generate_model_inputs(df_val)

    # Use subset of data for training
    training_subset_size = split_options["training_subset_size"]
    X = X.iloc[-training_subset_size:]
    y = y[-training_subset_size:]

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

    regressor = lgb.LGBMRegressor()

    # Search params
    use_cv = modeling_options["use_cv"]
    if use_cv:
        best_params = get_best_params(tuningparams, regressor, X, y, n_splits=2)
        pd.to_pickle(best_params, "best_params.pickle")
    else:
        if os.path.exists("best_params.pickle"):
            logger.info("Reading from best_params.pickle")
            best_params = pd.read_pickle("best_params.pickle")
        else:
            logger.info("Using default best parameters")
            best_params = modeling_options["best_params_default"]

    logger.info(best_params)
    best_regressor = lgb.LGBMRegressor(**best_params)
    best_regressor.fit(
        X, y, eval_set=[(X, y), (Xval, yval)], verbose=800, early_stopping_rounds=2000
    )
    # lgb.plot_importance(best_regressor)
    # lgb.plot_metric(best_regressor)
    # plt.show()

    ypred = best_regressor.predict(Xtest)
    ypred_train = best_regressor.predict(X)

    logger.info(
        "Train MAPE is %.3f%%"
        % (m.mean_absolute_percentage_error(y, ypred_train) * 100)
    )
    logger.info(
        "Test MAPE is %.3f%%" % (m.mean_absolute_percentage_error(ytest, ypred) * 100)
    )


if __name__ == "__main__":
    main()
