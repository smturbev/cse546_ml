if __name__ == "__main__":
    from coordinate_descent_algo import train  # type: ignore
else:
    from .coordinate_descent_algo import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")

    x_train = df_train.drop("ViolentCrimesPerPop", axis=1).values
    y_train = df_train["ViolentCrimesPerPop"].values
    x_test  = df_test.drop("ViolentCrimesPerPop", axis=1).values
    y_test  = df_test["ViolentCrimesPerPop"].values

    n, d = x_train.shape
    l = np.max(2 * abs(x_train.T @ (y_train - np.nanmean(y_train))))
    w0, b0 = train(x_train, y_train, _lambda=l)
    w0_test, b0_test = train(x_test, y_test, _lambda=l)

    num_nonzero = [np.count_nonzero(w0)]
    lambdas = [l]
    cols = ["agePct12t29", "pctWSocSec", "pctUrban", "agePct65up", "householdsize"]
    inds = np.array([ df_train.columns.get_loc(c) for c in cols ]) - 1
    w_d = np.array([[w0[i] for i in inds]])
    b_d = [b0]
    sq_er_train = [np.mean((x_train@w0-y_train+b0)**2)]
    sq_er_test  = [np.mean((x_test@w0_test-y_test+b0_test)**2)]
    while l >= 0.01:
        l = l/2
        w, b = train(x_train, y_train, _lambda=l, start_weight=w0)
        w_di = [w[i] for i in inds]
        sq_er_train = np.append(sq_er_train, np.mean((x_train@w-y_train+b)**2))
        sq_er_test  = np.append(sq_er_test,  np.mean((x_test @w-y_test +b)**2))
        num_nonzero = np.append(num_nonzero, np.count_nonzero(w))
        lambdas = np.append(lambdas, l)
        w_d = np.append(w_d, [w_di])
    w_d = np.reshape(w_d, (17,5))


    # (c)
    plt.figure()
    plt.plot(np.array(lambdas), np.array(num_nonzero), ".-")
    plt.xlabel("log($\lambda$)")
    plt.ylabel("num non zero weights")
    plt.xscale("log")
    plt.savefig("lambda_vs_num-nonzero.jpg")
    plt.close()

    # (d)
    plt.figure()
    for i in range(len(inds)):
        plt.plot(np.array(lambdas), np.array(w_d[:,i]), ".-", label=cols[i])
    plt.xlabel("log($\lambda$)")
    plt.ylabel("num non zero weights")
    plt.xscale("log")
    plt.legend()
    plt.savefig("lambda_vs_regularizationpaths.jpg")
    plt.close()

    # (e)
    plt.figure()
    plt.plot(np.array(lambdas), np.array(sq_er_train), ".-", label="training error")
    plt.plot(np.array(lambdas), np.array(sq_er_test), ".-", label="testing error")
    plt.xlabel("log($\lambda$)")
    plt.ylabel("num non zero weights")
    plt.xscale("log")
    plt.legend()
    plt.savefig("lambda_vs_sq-er.jpg")
    plt.close()

    print("done.")


def main_f(l=30):
    """code for problem A3 (f) looking at when lambda = 30."""
        # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")

    x_train = df_train.drop("ViolentCrimesPerPop", axis=1)
    y_train = df_train["ViolentCrimesPerPop"].values
    x_test  = df_test.drop("ViolentCrimesPerPop", axis=1).values
    y_test  = df_test["ViolentCrimesPerPop"].values

    n, d = x_train.shape
    w0, b0 = train(x_train.values, y_train, _lambda=l)
    w0_test, b0_test = train(x_test, y_test, _lambda=l)

    num_nonzero = np.count_nonzero(w0)
    sq_er_train = np.mean((x_train@w0-y_train+b0)**2)
    sq_er_test  = np.mean((x_test@w0_test-y_test+b0_test)**2)

    print("size w", w0.shape)
    print("num nonzero", num_nonzero)
    print("mean sq. error training", sq_er_train)
    print("mean sq. error test", sq_er_test)

    print("weights:\n", np.nonzero(w0))
    print("nonzero features:", x_train.columns[np.nonzero(w0)])
    print("MAX", np.max(w0), "ARG MAX", np.argmax(w0))
    print("largest feature (pos)", x_train.columns[np.argmax(w0)])
    print("largest feature (neg)", x_train.columns[np.argmin(w0)])

if __name__ == "__main__":
    main_f()
