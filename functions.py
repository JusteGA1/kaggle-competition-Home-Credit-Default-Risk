import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.model_selection import cross_val_score


class DataFrameImputer(TransformerMixin):
    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """

    def fit(self, X, y=None):

        self.fill = pd.Series(
            [
                X[c].value_counts().index[0]
                if X[c].dtype == np.dtype("O")
                else X[c].mean()
                for c in X
            ],
            index=X.columns,
        )

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


def get_feature_importance(features_list, feature_importance):
    df = pd.DataFrame(columns = ["feature", "importance"])
    for col,score in zip(features_list, feature_importance):
        if score > 0:
            df = df.append({"feature": col, "importance": score}, ignore_index=True)
    return df


def get_mean_scores(model, dataset_list, target):
    score_list = []
    for dataset in dataset_list:
        scores = cross_val_score(model, dataset, target, cv=3, scoring="roc_auc")
        scores_mean = round(scores.mean(), 3)
        score_list.append(scores_mean)
    return (
        "CV_all mean score all: " + str(score_list[0]),
        "CV_all mean score top lr: " + str(score_list[1]),
        "CV_all mean score main features: " + str(score_list[2]),
        "CV_all mean score all selected features: " + str(score_list[3])
    )


def percent_missing(df: pd.DataFrame) -> pd.Series:
    percent_nan = 100 * df.isnull().sum() / len(df)
    percent_nan = percent_nan[percent_nan > 0].sort_values()
    return percent_nan


def change_to_other(df, features, amount):
    for feature in features:
        mask = df[feature].map(df[feature].value_counts()) < amount
        df[feature] = df[feature].mask(mask, "Other")
    return df


def find_relation(dataframe, feature, target):
    df = dataframe.groupby([feature, target]).size().to_frame().reset_index()
    df = df.rename(columns={0: "total"})

    df["percentage"] = (df.total / df.total.sum()) * 100
    return df


def find_relation_in_groups(dataframe, feature, target):
    df = find_relation(dataframe, feature, target)
    df = df.groupby([feature, target]).agg({"total": "sum"})
    df = df.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))

    df = df.reset_index()

    return df


def clean_object_features(dataframe, columns_list):
    object_df = dataframe.select_dtypes(include="object")
    object_df_imputed = DataFrameImputer().fit_transform(object_df)

    df_objects_encoding = pd.get_dummies(object_df_imputed)

    return df_objects_encoding[columns_list]



