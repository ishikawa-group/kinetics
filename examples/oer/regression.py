import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn
import argparse

matplotlib.rcParams['backend'] = 'TkAgg'
np.random.seed(10)


def drop_duplicated_sample(df, key=None):
    if key is None:
        return df
    duplicated = df[key].apply(tuple).duplicated()
    df = df[~duplicated]
    return df


def regression(df, index_key=None, target_col="ads_energy", do_plot=True, method="lasso"):
    from sklearn.linear_model import LinearRegression, Lasso
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import mean_squared_error

    # drop duplicate, identified with index key (list)
    df = drop_duplicated_sample(df, key=index_key)

    # Determine columns to drop
    drop_cols = [target_col]
    if index_key is not None:
        drop_cols.append(index_key)

    x = df.drop(drop_cols, axis=1)
    y = df[target_col]

    cv = 10
    test_size = 1.0 / cv

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    scaler = StandardScaler()
    
    # Select method and parameter grid based on method parameter
    if method == "lasso":
        reg_method = Lasso()
        param_grid = {"reg__alpha": list(10**np.arange(-2, 2, 1.0))}
    elif method == "linear":
        reg_method = LinearRegression()
        param_grid = {}
    elif method == "random_forest":
        reg_method = RandomForestRegressor(random_state=10)
        param_grid = {
            "reg__n_estimators": [50, 100, 200],
            "reg__max_depth": [3, 5, 10, None],
        }
    else:
        raise ValueError(f"Unknown method: {method}")
    
    pipe = Pipeline([("scl", scaler), ("reg", reg_method)])
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv)
    grid.fit(x_train, y_train)

    # Print feature importance or coefficients based on method
    if method == "random_forest":
        print(pd.DataFrame({"name": x.columns,
                            "Importance": grid.best_estimator_.named_steps["reg"].feature_importances_}))
    else:
        print(pd.DataFrame({"name": x.columns,
                            "Coef": grid.best_estimator_.named_steps["reg"].coef_}))
    
    print("Training set score: {:.3f}".format(grid.score(x_train, y_train)))
    print("Test set score: {:.3f}".format(grid.score(x_test, y_test)))
    print("RMSE: {:.3f}".format(np.sqrt(mean_squared_error(y_test, grid.predict(x_test)))))

    if do_plot:
        fig, ax = plt.subplots(figsize=(6, 6))
        seaborn.regplot(x=grid.predict(x), y=y.values,
                        scatter_kws={"color": "navy", 'alpha': 0.3}, line_kws={"color": "navy"})
        ax.set_xlabel("Predicted value")
        ax.set_ylabel("True value")
        ax.set_title(f"Regression for {target_col} with {method}")
        fig.tight_layout()
        plt.show()
        # fig.savefig("plot.png")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default="output.csv",
                        help="filename containing data (json or csv)")
    parser.add_argument("--index_key", default=None,
                        help="column name to use as index key (default: 1st column)")
    parser.add_argument("--target_col", default=None,
                        help="column name to use as target variable (default: last column)")
    parser.add_argument("--method", default="lasso", choices=["lasso", "linear", "random_forest"],
                        help="regression method to use (default: lasso)")
    args = parser.parse_args()
    filename = args.filename

    # Extract file extension
    _, file_extension = os.path.splitext(filename)
    file_extension = file_extension.lower()

    # Read file based on extension
    if file_extension == ".json":
        df = pd.read_json(filename, orient="records", lines=True)
    elif file_extension == ".csv":
        df = pd.read_csv(filename)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    # Set defaults based on dataframe columns
    index_key = args.index_key if args.index_key is not None else df.columns[0]
    target_col = args.target_col if args.target_col is not None else df.columns[-1]

    plot = seaborn.pairplot(df)
    # plt.show()
    plt.savefig("pairplot.png")
    plt.close()

    regression(df, index_key=index_key, target_col=target_col, method=args.method)
