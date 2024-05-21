import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
import os
from typing import Tuple

ROOMS = ["living", "livingdown", "main", "studio"]


def read_data_to_dataframe(
    filepath: str = f"{os.path.dirname(os.path.realpath(__file__))}/data.pkl",
) -> pd.DataFrame:
    dataframe = pd.read_pickle(filepath)
    dataframe.reset_index(inplace=True)
    dataframe.dropna(inplace=True)

    return dataframe


def get_y(
    dataframe: pd.DataFrame, rooms: list[str] = ROOMS, measurement: str = "temperature"
) -> pd.DataFrame:
    """
    Get output y from dataframe. Default is temperature for each room.
    """

    keys = [f"{room}_{measurement}" for room in rooms]

    assert set(keys).issubset(set(dataframe.columns)), f"Keys {keys} not in dataframe"

    return dataframe.filter(items=keys)


def get_u(
    dataframe: pd.DataFrame, rooms: list[str] = ROOMS, measurement: str = "pump_action"
) -> pd.DataFrame:
    """
    Get input u from dataframe. Default is pump action and outdoor temperature.
    """

    keys = [f"{room}_{measurement}" for room in rooms]
    keys.append("outdoor_temperature")

    assert set(keys).issubset(set(dataframe.columns)), f"Keys {keys} not in dataframe"

    return dataframe.filter(items=keys)


def get_t(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Get time from dataframe.
    """

    return dataframe.filter(items=["time_abs [s]"])


def get_u_y_list_from_dataframe(
    dataframe: pd.DataFrame,
) -> Tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    y_array = get_y(dataframe).to_numpy()
    u_array = get_u(dataframe).to_numpy()

    y_list = np.split(y_array.T, y_array.T.shape[1], axis=1)
    u_list = np.split(u_array.T, u_array.T.shape[1], axis=1)
    return y_array, y_list, u_list


def get_nan_groups(dataframe: pd.DataFrame):
    # For each group, check if the group has more than one consecutive NaN
    mask = dataframe.isna().any(axis=1)
    nan_groups = []
    for _, g in mask.groupby((mask != mask.shift()).cumsum()):
        if g.sum() > 1:
            nan_groups.append(g.index.tolist())

    return nan_groups


def get_non_nan_groups(dataframe: pd.DataFrame):
    # For each group, check if the group has more than one consecutive NaN
    mask = dataframe.isna().any(axis=1)
    non_nan_groups = []
    for _, g in mask.groupby((mask != mask.shift()).cumsum()):
        if g.sum() == 0:
            non_nan_groups.append(g.index.tolist())

    return non_nan_groups


def plot_data(dataframe: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    u = get_u(dataframe)
    y = get_y(dataframe)

    groups = get_nan_groups(dataframe)

    figure, axes = plt.subplots(3, 1, sharex=True)
    u.filter(items=[f"{room}_pump_action" for room in ROOMS]).plot(ax=axes[0])
    y.plot(ax=axes[1])
    u.filter(items=["outdoor_temperature"]).plot(ax=axes[2])

    # Plot a patch for each g in groups
    for ax in axes:
        for g in groups:
            ax.axvspan(g[0], g[-1], color="red", alpha=0.5)

        ax.grid()

    axes[0].set_ylabel("u")
    axes[1].set_ylabel("y")
    axes[1].set_xlabel("Time")

    return figure, axes


def build_hankel(y: list[np.ndarray], i: int, s: int, bar_N: int) -> np.ndarray:
    """
    Build a block Hankel matrix based on input data y with starting index i, number of rows s, and number of columns bar_N.

    Parameters:
    - y: Input data, a list or 1D array of multi-dimensional arrays (e.g., vectors)
    - i: First index of the sequence that appears in the matrix
    - s: Number of rows
    - bar_N: Number of columns

    Returns:
    - Block Hankel matrix
    """

    if i + s + bar_N - 2 > len(y):
        raise ValueError(
            f"The provided parameters i= {i}, s={s}, barN = {bar_N} lead to indices outside the range of y."
        )

    # Element dimensionality
    dim = y[0].shape

    # Initialize the block Hankel matrix
    H = np.zeros((s * dim[0], bar_N * dim[1]))

    for row in range(s):
        for col in range(bar_N):
            H[row * dim[0] : (row + 1) * dim[0], col * dim[1] : (col + 1) * dim[1]] = y[
                i + row + col
            ]

    assert H.shape == (s * dim[0], bar_N * dim[1])

    return H


def test_build_hankel() -> None:
    y = [i * np.ones((3, 1)) for i in range(10)]

    # Example:
    i = 2
    s = 4
    bar_N = 5

    print(y)

    Y = build_hankel(y, i, s, bar_N)
    print(Y)

    return None

def build_linear_n_step_prediction_model(
    u_list: list[np.ndarray],
    y_list: list[np.ndarray],
    n_init: int,
    n_pred: int,
    first_sample: int,
    n_data: int,
    future_error_from_full_hankel = False,
    plot_w = False,
) -> np.ndarray:
    """
    Build linear n-step prediction model using Hankel matrices

    Parameters
    ----------
    u_list : list[np.ndarray]
        List of input data
        y_list : list[np.ndarray]
        List of output data
        n_init : int
        Number of samples to use as initial condition
        n_pred : int
        Number of samples to predict
        first_sample : int
        First sample to use in Hankel matrices
        n_data : int
        Number of samples in Hankel matrices (essentially size of training data)
        future_error_from_full_hankel : bool 
        plot_w : bool
        plot the w distribution

    Returns
    -------
    np.ndarray
        Phi matrix. The model is Y_ip_Nf_barN = Phi @ [U_i_Np_barN; Y_i_Np_barN; U_ip_Nf_barN] or easier to read Yhat = Phi @ [Up; Yp; Uf]

    # TODO Add multiple output identified line-by-line for causality
    # TODO Add regularization
    """
    Y_i_Np_barN = build_hankel(y_list, first_sample, n_init, n_data)
    U_i_Np_barN = build_hankel(u_list, first_sample, n_init, n_data)
    Y_ip_Nf_barN = build_hankel(y_list, first_sample + n_init, n_pred, n_data)
    U_ip_Nf_barN = build_hankel(u_list, first_sample + n_init, n_pred, n_data)
    # Solve min ||Y_ip_Nf_barN - Phi Z_barN||_F^2
    Z_barN = np.vstack((U_i_Np_barN, Y_i_Np_barN, U_ip_Nf_barN))
    Phi = np.linalg.lstsq(Z_barN.T, Y_ip_Nf_barN.T, rcond=None)[0].T
    y_hat = Phi @ Z_barN
    error = Y_ip_Nf_barN - y_hat
    error_list = []
    for row in error.T:
        reshaped_array = row.reshape(4, 1)
        error_list.append(reshaped_array)
    print(np.shape(error_list), np.shape(y_list))
    # Solve min ||Y_ip_Nf_barN - Phi Z_barN||_F^2
    first_sample_error = 1
    error_p_hankel = build_hankel(error_list, first_sample_error, n_init, n_data-n_init-1)
    error_f_hankel = build_hankel(error_list, n_init+first_sample_error, n_pred, n_data-n_init-1)
    if future_error_from_full_hankel:
        U_i_Np_barN = build_hankel(u_list, first_sample, n_init, n_data-n_init-1)
        U_ip_Nf_barN = build_hankel(u_list, first_sample + n_init, n_pred, n_data-n_init-1)
        error_p_hankel = np.vstack((U_i_Np_barN, error_p_hankel, U_ip_Nf_barN))
    ePhi = np.linalg.lstsq(error_p_hankel.T, error_f_hankel.T, rcond=None)[0].T
    
    error_hat = ePhi @ error_p_hankel
    e_error = error[:, 25:] - error_hat
    
    if plot_w:
        plt.figure()
        for i in range(4):
            plt.subplot(4,1,i+1)
            plt.hist(e_error[i], bins=30, alpha=0.75, color='blue', edgecolor='black')
        plt.show()
    
    return Phi, ePhi

def main():
    dataframe = read_data_to_dataframe()

    #  We sample at 5 minutes and want to predict n_pred_hours ahead
    # n_pred_hours = 12
    # n_pred = n_pred_hours * 60 // 5

    # For now, we predict 1 step ahead
    n_pred = 1

    # We sample at 5 minutes and use n_init_hours hours of data for initial condition
    n_init_hours = 2
    n_init = n_init_hours * 60 // 5

    # Columns in Hankel matrix. We want 30 days of of training data sampled at 5 minutes
    n_train_hours = 10 * 24
    n_train = n_train_hours * 60 // 5

    # Represent data as list of column vectors
    y_array, y_list, u_list = get_u_y_list_from_dataframe(dataframe)

    step = 6 * 60 // 5

    # nu = u_list[0].shape[0]
    ny = y_list[0].shape[0]

    n_iterations = len(y_list) // step

    print(f"Number of iterations: {n_iterations}")

    # Samples to start building test predictions from, i.e. those not included in the Hankel matrices
    N = n_train + n_init + n_pred - 1

    error_list = []
    plot_period = 100
    y_hat_array = np.zeros((plot_period, 4))
    y_true_array = np.zeros((plot_period, 4))
    error_array = np.zeros((n_init, 4))
    for row in error_array:
        reshaped_array = row.reshape(4, 1)
        error_list.append(reshaped_array)
    j = 0
    for iteration, i in enumerate(range(1, len(y_list), step)):
        start_time = time.time()
        Phi, ePhi = build_linear_n_step_prediction_model(
            u_list, y_list, n_init, n_pred, i, n_train, False, False,
        )
        print(np.shape(Phi))
        # Determine
        k = i + N
        u_p = build_hankel(u_list, k - n_init + 1, n_init, 1)
        y_p = build_hankel(y_list, k - n_init + 1, n_init, 1)
        u_f = build_hankel(u_list, k + 1, n_pred, 1)
        error_past = build_hankel(error_list, 0, n_init, 1)

        # Reshape yhat to (Nf, ny)
        yhat = Phi @ np.vstack((u_p, y_p, u_f)) + ePhi @ error_past
        yhat = yhat.reshape(n_pred, ny)
        y_true = y_array[k + 1 : k + n_pred + 1, :]
        error = (y_true - yhat).reshape(4,1)
        error_list.append(error) 
        error_list.pop(0)
        y_hat_array[j,:] = yhat 
        y_true_array[j,:] = y_true 
        j = j+1
        if j==plot_period:
            plt.figure()
            for i in range(4):
                plt.subplot(4,1,i+1)
                plt.plot(y_true_array[:,i], color='red')
                plt.plot(y_hat_array[:,i], color='blue')
            plt.show()
            j = 0
        # plt.figure()
        # for i_ax in range(1, ny + 1):
        #     plt.subplot(ny, 1, i_ax)
            
        #     plt.plot(y_true[:, i_ax - 1], label="True")
        #     plt.plot(yhat[:, i_ax - 1], label="Predicted")
        #     plt.legend()
        # plt.show()

        end_time = time.time()
        print(
            f"Iteration {iteration} of {n_iterations}. Time: {end_time - start_time:.2f} s"
        )
if __name__ == "__main__":
    main()
