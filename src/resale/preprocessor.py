from sklearn.preprocessing import OrdinalEncoder


class Preprocessor:
    def __init__(self, label_column=None, categorical_columns=None, cols_to_keep=None):
        self.label_column = label_column
        self.categorical_columns = categorical_columns
        self.cols_to_keep = cols_to_keep
        self._is_fitted = False
        self.ordinal_encs = {}

    def generate_model_inputs(self, df, train=False):
        X = df.drop(columns=self.label_column)
        y = df[self.label_column].values

        X_feat = self._do_feature_engineering(X, train=True)
        return X_feat, y

    def _do_feature_engineering(self, df, train=False):

        df_out = df.copy()

        if not self._is_fitted and not train:
            raise ValueError("Preprocessor is not fitted, cannot use train=False")

        for c in self.categorical_columns:
            if train:
                self.ordinal_encs[c] = OrdinalEncoder()
                self.ordinal_encs[c].fit(df_out[c].values.reshape(-1, 1))

            df_out[c] = (
                self.ordinal_encs[c]
                .transform(df_out[c].values.reshape(-1, 1))
                .astype(int)
            )

            # Set categorical type for LightGBM
            df_out[c] = df_out[c].astype("category")

        if train:
            self._is_fitted = True

        df_out = df_out[self.cols_to_keep]
        return df_out
