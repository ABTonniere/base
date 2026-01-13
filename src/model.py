def xgboost_trained(X_train, y_train):

    xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss",
    )
    xgb_model.fit(X_train, y_train)

    return xgb_model