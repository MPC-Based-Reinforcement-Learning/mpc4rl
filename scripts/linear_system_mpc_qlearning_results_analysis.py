import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scripts.linear_system_mpc_qlearning import get_res_dir


if __name__ == "__main__":

    dataframe = pd.read_csv(f"{get_res_dir()}/data.csv")

    print(dataframe.keys())

    plt.figure(1)
    plt.subplot(211)
    dataframe.plot(y=["cost"], ax=plt.gca(), grid=True)
    plt.subplot(212)
    dataframe.plot(y=["td_error"], ax=plt.gca(), grid=True)

    nrows = 5
    plt.figure(2)
    dataframe.plot(y=[f"A_{i}" for i in range(4)], ax=plt.subplot(nrows, 1, 1), grid=True)
    dataframe.plot(y=[f"B_{i}" for i in range(2)], ax=plt.subplot(nrows, 1, 2), grid=True)
    dataframe.plot(y=[f"b_{i}" for i in range(2)], ax=plt.subplot(nrows, 1, 3), grid=True)
    dataframe.plot(y=[f"V_{i}" for i in range(1)], ax=plt.subplot(nrows, 1, 4), grid=True)
    dataframe.plot(y=[f"f_{i}" for i in range(3)], ax=plt.subplot(nrows, 1, 5), grid=True)

    plt.show()
