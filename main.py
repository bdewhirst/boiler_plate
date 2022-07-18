import sklearn

import utils.common_utils as u
import model.model as model


def do_lm(conn) -> None:
    all_training = u.run_sqlite_query(conn=conn, table_name="train")
    to_drop = ["PassengerId", "Name", "Ticket", "Cabin", ]
    to_dummy = ["Sex", "Embarked", ]
    fix_nan = ["Age", ]
    all_train_cleaned = model.prep_data(df=all_training, cols_to_drop=to_drop, cols_to_dummy=to_dummy, cols_w_nan=fix_nan)
    train_on = ["pclass", "age", "sibsp", "parch", "fare", "sex_male", "embarked_c",
                "embarked_q"]  # not parsimonius...
    xs, y = model.split_xs_and_ys(df=all_train_cleaned, x_cols=train_on, )
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(xs, y, test_size=0.25)
    del all_train_cleaned, xs, y

    trained_linear_model = model.fit_lin_reg(xs=x_train, y=y_train)

    score = model.score_fit(model=trained_linear_model, x_test=x_test, y_test=y_test, )
    # more scoring stuff here
    # graphs here

    test= u.run_sqlite_query(conn=conn, table_name="test")
    result= model.fit_unknown_data(test=test, model_fit=trained_linear_model)
    print(result.head()) # tmp
    # more here


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
    finally:
        if conn:
            conn.close()
    print("done")  # with database retrieval")


if __name__ == '__main__':
    main()