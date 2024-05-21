import pandas as pd
import numpy as np
# import matplotlib
# matplotlib.use('macosx') 
from matplotlib import pyplot as plt
import time
import os
from typing import Tuple
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
import matplotlib.gridspec as gridspec
from collections import deque

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

def test_correlaton(data:np.ndarray):
    # Plot autocorrelation
    num_vars = data.shape[0]

    # Create a grid for plotting
    fig = plt.figure(figsize=(12, 4 * num_vars))
    gs = gridspec.GridSpec(num_vars, 2)

    for i in range(num_vars):
        # Autocorrelation plot for each variable
        ax = fig.add_subplot(gs[i, 0])
        plot_acf(data[:, i], ax=ax, title=f'Autocorrelation for Variable {i+1}')
        
        # Durbin-Watson statistic for each variable
        dw_stat = durbin_watson(data[:, i])
        print(f'Durbin-Watson statistic for Variable {i+1}: {dw_stat}')
        
        # Adding a subplot for histogram (or any other plot for distribution checking)
        ax = fig.add_subplot(gs[i, 1])
        ax.hist(data[:, i], bins=20, alpha=0.7)
        ax.set_title(f'Histogram for Variable {i+1}')

    plt.tight_layout()
    plt.show()

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

def estimate_density(data: np.ndarray, plot_dist):
    # Number of variables
    num_vars = data.shape[0]
    # Step 2: Estimate Gaussian density for each variable and sample from it
    samples = []
    dist = np.zeros((num_vars,2))
    for i in range(num_vars):
        # Current variable data
        variable_data = data[i]
        # Fit a normal distribution to the data:
        dist[i,0], dist[i,1] = norm.fit(variable_data)
    #dist[:,1] = dist[:,1] - [.18,0.14,0.1,0.1]
    if plot_dist:
        fig, axes = plt.subplots(num_vars, 1, figsize=(8, 6 * num_vars))
        for i in range(num_vars):
            # Plot the histogram
            mu = dist[i,0]
            std = dist[i,1]
            axes[i].hist(variable_data, bins=100, density=True, alpha=0.6, color='g')

            # Plot the PDF.
            xmin, xmax = axes[i].get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            axes[i].plot(x, p, 'k', linewidth=2)
            title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
            axes[i].set_title(title)

            # Sample from the distribution
            sample = norm.rvs(mu, std, size=100)
            samples.append(sample)

            # Show samples on the plot for visual verification
            axes[i].scatter(sample, np.random.normal(0, 0.01, size=sample.size), color='r', zorder=10, label='Samples')
            axes[i].legend()

        plt.tight_layout()
        plt.show()
    return dist

def build_linear_n_step_prediction_model(
    u_list: list[np.ndarray],
    y_list: list[np.ndarray],
    n_init: int,
    n_pred: int,
    first_sample: int,
    n_data: int,
    future_error_from_full_hankel: bool ,
    plot_error: bool,
    plot_dist: bool,
    correlation_test: bool,
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
    
#     n = 13  # For example, average of past 10 vectors   
#     error_filter = np.zeros((4, error.shape[1] - n + 1))
#   # Compute the rolling average using a sliding window approach
#     for i in range(error_filter.shape[1]):
#         error_filter[:, i] = error[:, i:i+n].mean(axis=1)
#     y_hat = y_hat[:,n-1:] + error_filter
#     e_error = Y_ip_Nf_barN[:,n-1:] - y_hat
    

    ## Commented from here
    error_list = []
    for row in error.T:
        reshaped_array = row.reshape(4, 1)
        error_list.append(reshaped_array)
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

    # e_error_list = []
    # for row in e_error.T:
    #     reshaped_array = row.reshape(4, 1)
    #     e_error_list.append(reshaped_array)
    # # Solve min ||Y_ip_Nf_barN - Phi Z_barN||_F^2
    # first_sample_error = 1
    # e_error_p_hankel = build_hankel(e_error_list, first_sample_error, n_init, n_data-2*n_init-2)
    # e_error_f_hankel = build_hankel(e_error_list, n_init+first_sample_error, n_pred, n_data-2*n_init-2)
    # if future_error_from_full_hankel:
    #     U_i_Np_barN = build_hankel(u_list, first_sample, n_init, n_data-2*n_init-2)
    #     U_ip_Nf_barN = build_hankel(u_list, first_sample + n_init, n_pred, n_data-2*n_init-2)
    #     e_error_p_hankel = np.vstack((U_i_Np_barN, e_error_p_hankel, U_ip_Nf_barN))
    # wPhi = np.linalg.lstsq(e_error_p_hankel.T, e_error_f_hankel.T, rcond=None)[0].T
    # e_error_hat = wPhi @ e_error_p_hankel
    # e_e_error = e_error[:, 25:] - e_error_hat
    # print(e_e_error)
    
    
    
    if plot_error:
        plt.figure()
        for i in range(4):
            plt.subplot(4,1,i+1)
            plt.hist(e_error[i], bins=30, alpha=0.75, color='blue', edgecolor='black')
        plt.show()
    if correlation_test:
        test_correlaton(e_error)
    dist = estimate_density(e_error, plot_dist)
    return Phi, dist

def build_simulator(
    u_list: list[np.ndarray],
    y_list: list[np.ndarray],
    n_init: int,
    n_pred: int,
    first_sample: int,
    n_data: int,
    n_initialize: int

):
    Phi, ePhi = build_linear_n_step_prediction_model(
        u_list, y_list, n_init, n_pred, first_sample, n_data, False, False,
    )
    return Phi, ePhi

def step(state, action):

    return next_state

def main():
    dataframe = read_data_to_dataframe()
    #  We sample at 5 minutes and want to predict n_pred_hours ahead
    n_pred = 1

    # We sample at 5 minutes and use n_init_hours hours of data for initial condition
    n_init_hours = 2
    n_init = n_init_hours * 60 // 5

    # Columns in Hankel matrix. We want 30 days of of training data sampled at 5 minutes
    n_train_hours = 132 * 24 
    n_train = n_train_hours * 60 // 5

    # Represent data as list of column vectors
    y_array, y_list, u_list = get_u_y_list_from_dataframe(dataframe)

    ny = y_list[0].shape[0]
    
    #SETTINGS
    #index of the first point in the dataset u want to use
    start_index = 1
    # plot the distribution of the error of the erros
    plot_dist = False
    # generative stochasticity in the simulator bu sampling from error of error distribution
    sample_from_error = False
    # test if the error of the error dist is IID
    correlation_test = False
    #no of simulation steps
    n_steps = 10000 
    # forgetting factor for error term
    tau =100

    start_time = time.time()

    Phi, dist_error = build_linear_n_step_prediction_model(
        u_list, y_list, n_init, n_pred, start_index, n_train, False, False, plot_dist, correlation_test,
    )

#SIMULATE
    #set values for heatpump and outdoor temperature if needed
    set_inputs = False
    if set_inputs:
        for arr in u_list:
            arr[0:3] = 0
            arr[4] = -10

    y_hat_array = np.zeros((n_steps, 4))
    y_true_array = np.zeros((n_steps, 4))
    y_sim_list = y_list[0:n_init]
    sum_error = np.zeros((4,1))
    for j in range (n_steps):
        # Determine
        u_p = build_hankel(u_list, j, n_init, 1)
        u_f = build_hankel(u_list, j + n_init, n_pred, 1)
        y_p = build_hankel(y_sim_list, j, n_init, 1)
        
        # Reshape yhat to (Nf, ny)
        yhat = Phi @ np.vstack((u_p, y_p, u_f))#
        if sample_from_error:
            error_sample = norm.rvs(dist_error[:,0].flatten(), dist_error[:,1].flatten())
            # Convert deque to a NumPy array to sum the vectors
            sum_error = sum_error* np.exp(-j/tau) + error_sample.reshape(4,1) #np.average(np.array(list(error_past)), axis=0)
            print(sum_error)
            yhat = yhat + sum_error #+ ePhi @ error_past
        y_sim_list.append(yhat)
        # y_sim_list.pop(0)
        yhat = yhat.reshape(n_pred, ny)
        y_true = y_array[j + n_init + 1 : j + n_init + n_pred + 1, :]
        y_hat_array[j,:] = yhat 
        y_true_array[j,:] = y_true 
    plt.figure()
    for i in range(4):
        plt.subplot(4,1,i+1)
        #plt.plot(y_true_array[:,i], color='red')
        plt.plot(y_hat_array[:,i], color='blue')
    plt.show()

    end_time = time.time()
    print(
        f"Time: {end_time - start_time:.2f} s"
    )
if __name__ == "__main__":
    main()
