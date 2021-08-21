import pandas as pd
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMModel, LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

RANDOM = 42

xgb_params = dict(
    model__max_depth=list(range(2, 6)),
    model__min_child_weight=list(range(1, 20)),
    model__subsample=[0.5, 0.6, 0.7, 0.8, 0.9],
    model__gamma=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    model__colsample_bytree=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    model__reg_lambda=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4],
    model__reg_alpha=[1e-5, 1e-2, 0.1, 1, 100],
    model__learning_rate=[0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.13, 0.16, 0.19],
    model__n_estimators=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                         2000],
)

lgb_params = dict(
    model__max_depth=list(range(2, 6)),
    model__num_leaves=[6, 8, 12, 16, 32, 48, 100, 130, 150, 180, 200],
    model__min_data_in_leaf=[5, 10, 25, 50, 75, 100, 150],
    model__feature_fraction=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    model__subsample=[0.5, 0.6, 0.7, 0.8, 0.9],
    model__learning_rate=[0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.13, 0.16, 0.19],
    model__n_estimators=[200, 300, 500, 700],
)

cbc_params = dict(
    model__learning_rate=[0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.13, 0.16, 0.19],
    model__max_depth=list(range(2, 6)),
    model__subsample=[0.5, 0.6, 0.7, 0.8, 0.9],
    model__n_estimators=[200, 300, 500, 700]
)


def check_dataframes(dataframes, models):
    output = pd.DataFrame()
    for dataframe in dataframes:
        for model in models:
            result = check_model_results(dataframe, model)
            output = output.append(result, ignore_index=True)
    return output


def check_model_results(
    dataframe: pd.DataFrame,
    model
) -> dict:
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])

    train_sample = dataframe.sample(frac=0.01, replace=True, random_state=42)

    X = train_sample[train_sample.columns[2:]].values
    y = train_sample["TARGET"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM
    )

    if model.name is "CatBoost":
        pipe_rs = RandomizedSearchCV(
            pipe, cbc_params, n_iter=15, cv=3, verbose=1, n_jobs=5,
            random_state=RANDOM)
        pipe_rs.fit(X_train, y_train)
        model_with_params = select_cbc_model_params(pipe_rs)

    elif model.name is "XGB":
        pipe_rs = RandomizedSearchCV(
            pipe, xgb_params, n_iter=15, cv=3, verbose=1, n_jobs=5,
            random_state=RANDOM)
        pipe_rs.fit(X_train, y_train)
        model_with_params = select_xgb_model_params(pipe_rs)

    else:
        pipe_rs = RandomizedSearchCV(
            pipe, lgb_params, n_iter=15, cv=3, verbose=1, n_jobs=5,
            random_state=RANDOM)
        pipe_rs.fit(X_train, y_train)
        model_with_params = select_lgbm_model_params(pipe_rs)

    start = time.time()
    model_with_params.fit(X_train, y_train)
    end = time.time()

    y_pred_train = model_with_params.predict_proba(X_train)
    y_pred = model_with_params.predict_proba(X_test)

    return {
        "model": model,
        "dataframe": dataframe.name,
        "best_params": pipe_rs.best_params_,
        "training score": round(roc_auc_score(y_train, y_pred_train[:, 1]), 3),
        "validation score": round(roc_auc_score(y_test, y_pred[:, 1]), 3),
        "time": round(end - start, 4),
    }


def select_lgbm_model_params(pipe_rs):
    return LGBMClassifier(random_state=RANDOM,
                          objective="binary",
                          verbose=-1,
                          max_depth=pipe_rs.best_params_['model__max_depth'],
                          num_leaves=pipe_rs.best_params_['model__num_leaves'],
                          min_data_in_leaf=pipe_rs.best_params_[
                              'model__min_data_in_leaf'],
                          feature_fraction=pipe_rs.best_params_[
                              'model__feature_fraction'],
                          subsample=pipe_rs.best_params_['model__subsample'],
                          learning_rate=pipe_rs.best_params_[
                              "model__learning_rate"],
                          n_estimators=pipe_rs.best_params_[
                              "model__n_estimators"]
                          )


def select_xgb_model_params(pipe_rs):
    return XGBClassifier(random_state=RANDOM,
                         verbosity=0,
                         nthread=4,
                         max_depth=pipe_rs.best_params_['model__max_depth'],
                         min_child_weight=pipe_rs.best_params_[
                             'model__min_child_weight'],
                         subsample=pipe_rs.best_params_['model__subsample'],
                         gamma=pipe_rs.best_params_['model__gamma'],
                         colsample_bytree=pipe_rs.best_params_[
                             'model__colsample_bytree'],
                         reg_lambda=pipe_rs.best_params_['model__reg_lambda'],
                         reg_alpha=pipe_rs.best_params_['model__reg_alpha'],
                         learning_rate=pipe_rs.best_params_[
                             "model__learning_rate"],
                         n_estimators=pipe_rs.best_params_[
                             "model__n_estimators"]
                         )


def select_cbc_model_params(pipe_rs):
    return CatBoostClassifier(random_state=RANDOM,
                              verbose=0,
                              learning_rate=pipe_rs.best_params_[
                                  "model__learning_rate"],
                              n_estimators=pipe_rs.best_params_[
                                  "model__n_estimators"],
                              max_depth=pipe_rs.best_params_[
                                  'model__max_depth'],
                              subsample=pipe_rs.best_params_['model__subsample']
                              )
