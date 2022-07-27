import sklearn
import pandas as pd

import utils.common_utils as u
import model.model as model


def ready_training(conn) -> pd.DataFrame:
    all_training = u.run_sqlite_query(conn=conn, table_name="train")
    to_drop = ["PassengerId", "Name", "Ticket", "Cabin", ]
    to_dummy = ["Sex", "Embarked", ]
    fix_nan = ["Age", ]
    all_train_cleaned = model.prep_data(df=all_training, cols_to_drop=to_drop, cols_to_dummy=to_dummy, cols_w_nan=fix_nan)
    return all_train_cleaned


def do_test_train_split(df: pd.DataFrame, train_on: list, test_size: float=0.25) -> tuple:
    xs, y = model.split_xs_and_ys(df=df, x_cols=train_on, )
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(xs, y, test_size=test_size)
    return x_train, x_test, y_train, y_test


def do_lm(conn) -> None:
    all_train_cleaned = ready_training(conn)
    # n.b.: parameter selection is not parsimonious
    train_on = ["pclass", "age", "sibsp", "parch", "fare", "sex_male", "embarked_c", "embarked_q"]
    x_train, x_test, y_train, y_test= do_test_train_split(df=all_train_cleaned, train_on=train_on, )

    trained_linear_model = model.fit_lin_reg(xs=x_train, y=y_train)
    score = model.score_fit(model=trained_linear_model, x_test=x_test, y_test=y_test, )
    print(score)

    # try statsmodels linear model, as it provides more details on the fit
    model.do_statsmodels_lm(xs=x_train, y=y_train)  # or on raw xs, ys. It is probably doing its own holdout, or equiv.

    # more scoring stuff here
    # graphs here
    model.strawman_plot(xs=x_train, y=y_train, cols=["pclass", "age", "sibsp", "parch", "fare", "sex_male", "embarked_c", "embarked_q", "survived"])

    test= u.run_sqlite_query(conn=conn, table_name="test")
    result= model.fit_unknown_data(test=test, model_fit=trained_linear_model)
    print(result.head()) # tmp
    # more here, and/or break out model evaluation/tuning


def do_xgb(conn) -> None:
    all_train_cleaned = ready_training(conn)
    train_on = ["pclass", "age", "sibsp", "parch", "fare", "sex_male", "embarked_c", "embarked_q"]
    x_train, x_test, y_train, y_test= do_test_train_split(df=all_train_cleaned, train_on=train_on, )
    # note: following code should be in model.model...
    pass  # TODO


def main() -> None:
    """
    Main execution path of function
    :return: nothing
    """
    try:
        conn = u.sqlite_connect()
        # train = u.run_sqlite_query(conn=conn, table_name="train")
        # test = u.get_table(conn=conn, table_name="test")
        do_lm(conn=conn)
        do_xgb(conn=conn)
    finally:
        if conn:
            conn.close()
    print("done")  # with database retrieval")


if __name__ == '__main__':
    main()