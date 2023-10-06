import numpy as np
from matplotlib import pyplot as plt


def f(p):
    A, theta = p
    return np.array([A * np.cos(theta), A * np.sin(theta)])


def df_dp(p):
    A, theta = p
    return np.array([[np.cos(theta), -A * np.sin(theta)], [np.sin(theta), A * np.cos(theta)]])


p0 = np.array([1.0, np.pi / 2])

# Construct p_test for independent variation of A and theta
s = np.arange(0.8, 1.2, 0.001)
p_test_A = np.vstack((s * p0[0], np.full_like(s, p0[1]))).T
p_test_theta = np.vstack((np.full_like(s, p0[0]), s * p0[1])).T

p_test = [p_test_A, p_test_theta]

# p_test = np.array([s * p0 for s in np.arange(0.8, 1.2, 0.001)])
# p_test = np.array([s * p0 for s in np.arange(0.8, 1.2, 0.01), p0[1]])


for p in p_test:
    xf = np.zeros((p.shape[0], 2))
    dxf_dp = np.zeros((p.shape[0], 2, 2))
    for i in range(p.shape[0]):
        xf[i, :] = f(p[i, :])
        dxf_dp[i, :, :] = df_dp(p[i, :])

    dxf_grad = np.zeros_like(dxf_dp)
    dxf_grad[:, 0, 0] = np.gradient(xf[:, 0], p[:, 0], axis=0)
    dxf_grad[:, 0, 1] = np.gradient(xf[:, 0], p[:, 1], axis=0)
    dxf_grad[:, 1, 0] = np.gradient(xf[:, 1], p[:, 0], axis=0)
    dxf_grad[:, 1, 1] = np.gradient(xf[:, 1], p[:, 1], axis=0)

    s = np.arange(0.8, 1.2, 0.01)

    nparam = 2

    for i_param in range(nparam):
        fig, ax = plt.subplots(nrows=2, ncols=2)
        ax[0, 0].plot(p[:, i_param], xf[:, 0], label="x")
        ax[1, 0].plot(p[:, i_param], xf[:, 1], label="y")
        ax[0, 1].plot(p[:, i_param], dxf_dp[:, 0, i_param], label="dx")
        ax[1, 1].plot(p[:, i_param], dxf_dp[:, 1, i_param], label="dy")
        ax[0, 1].plot(p[:, i_param], dxf_grad[:, 0, i_param], "-.", label="dx")
        ax[1, 1].plot(p[:, i_param], dxf_grad[:, 1, i_param], "-.", label="dy")


plt.show()
