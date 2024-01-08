import os
from acados_template.acados_ocp_solver import ocp_generate_external_functions
import numpy as np
from acados_template import (
    AcadosOcp,
    AcadosOcpSolver,
    AcadosOcpConstraints,
    AcadosOcpCost,
    AcadosOcpDims,
    AcadosModel,
    AcadosOcpOptions,
)

import casadi as cs

from rlmpc.common.mpc import MPC

import matplotlib.pyplot as plt

from rlmpc.mpc.cartpole.common import Config, ModelParams, define_parameter_values

from rlmpc.common.integrator import ERK4

from rlmpc.mpc.nlp import LagrangeMultiplierMap, NLP, update_nlp, find_nlp_entry_expr_dependencies, build_nlp


def define_acados_model(ocp: AcadosOcp, config: dict) -> AcadosModel:
    # Check if ocp is an instance of AcadosOcp
    if not isinstance(ocp, AcadosOcp):
        raise TypeError("ocp must be an instance of AcadosOcp")

    name = config["model"]["name"]

    # set up states & controls
    s = cs.SX.sym("x")
    s_dot = cs.SX.sym("x_dot")
    theta = cs.SX.sym("theta")
    theta_dot = cs.SX.sym("theta_dot")

    x = cs.vertcat(s, s_dot, theta, theta_dot)

    F = cs.SX.sym("F")
    u = cs.vertcat(F)

    x_dot = cs.SX.sym("xdot", 4, 1)

    # algebraic variables
    z = None

    # parameters
    p_sym = []

    model_params = ModelParams.from_dict(config["model"]["params"])

    # Set up parameters to nominal values
    p = {key: param["value"] for key, param in model_params.to_dict().items()}

    parameter_values = []
    # Set up parameters to symbolic variables if not fixed
    for key, param in model_params.to_dict().items():
        if not param["fixed"]:
            p_sym += [cs.SX.sym(key)]
            p[key] = p_sym[-1]
            parameter_values += [param["value"]]

    p_sym = cs.vertcat(*p_sym)
    parameter_values = np.array(parameter_values)

    # Define model dynamics
    cos_theta = cs.cos(theta)
    sin_theta = cs.sin(theta)
    temp = (u + p["m"] * theta_dot**2 * sin_theta) / (p["m"] + p["M"])

    theta_ddot = (p["g"] * sin_theta - cos_theta * temp) / (p["l"] * (4.0 / 3.0 - p["m"] * cos_theta**2 / (p["m"] + p["M"])))

    f_expl = cs.vertcat(
        s_dot,
        temp - p["m"] * theta_ddot * cos_theta / (p["m"] + p["M"]),  # x_ddot
        theta_dot,
        (p["g"] * sin_theta - cos_theta * temp) / (p["l"] * (4.0 / 3.0 - p["m"] * cos_theta**2 / (p["m"] + p["M"]))),
    )

    f_impl = x_dot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = x_dot
    model.p = p_sym
    model.u = u
    model.z = z
    model.name = name

    return model


def define_acados_dims(config: Config) -> AcadosOcpDims:
    dims = AcadosOcpDims()

    for key, val in config.items():
        hasattr(dims, key), f"AcadosOcpDims does not have attribute {key}"

        # Set the attribute, assuming the value is correct
        # TODO: Add validation for the value here
        setattr(dims, key, val)

    return dims


def define_acados_cost(config: dict) -> AcadosOcpCost:
    cost = AcadosOcpCost()
    for key, val in config.items():
        assert hasattr(cost, key), f"AcadosOcpCost does not have attribute {key}"

        # Set the attribute, assuming the value is correct
        # TODO: Add validation for the value here
        if isinstance(val, list):
            setattr(cost, key, np.array(val))
        if isinstance(val, str):
            setattr(cost, key, val)

    return cost


def define_acados_constraints(config: dict) -> AcadosOcpConstraints:
    constraints = AcadosOcpConstraints()
    for key, val in config.items():
        hasattr(constraints, key), f"AcadosOcpConstraints does not have attribute {key}"

        if isinstance(val, list):
            setattr(constraints, key, np.array(val))
        if isinstance(val, str):
            setattr(constraints, key, val)

    return constraints


def define_acados_ocp_options(config: dict) -> AcadosOcpOptions:
    ocp_options = AcadosOcpOptions()
    for key, val in config.items():
        hasattr(ocp_options, key), f"AcadosOcpOptions does not have attribute {key}"

        setattr(ocp_options, key, val)

    return ocp_options


class AcadosMPC(MPC):
    """docstring for CartpoleMPC."""

    _parameters: np.ndarray
    ocp_solver: AcadosOcpSolver
    nlp: NLP
    idx: dict
    muliplier_map: LagrangeMultiplierMap

    def __init__(self, config: Config, build: bool = True):
        super().__init__()

        ocp = AcadosOcp()

        ocp.model = define_acados_model(ocp=ocp, config=config)

        ocp.model.disc_dyn_expr = ERK4(
            cs.Function("ode", [ocp.model.x, ocp.model.u, ocp.model.p], [ocp.model.f_expl_expr]),
            ocp.model.x,
            ocp.model.u,
            ocp.model.p,
            # config.ocp_options.tf / config.dimensions.N / config.ocp_options.sim_method_num_stages,
            config["ocp_options"]["tf"] / config["dimensions"]["N"] / config["ocp_options"]["sim_method_num_stages"],
        )

        ocp.parameter_values = define_parameter_values(config=config)

        ocp.constraints = define_acados_constraints(config=config["constraints"])

        ocp.dims = define_acados_dims(config=config["dimensions"])

        ocp.cost = define_acados_cost(config=config["cost"])

        ocp.model.cost_y_expr_0 = cs.vertcat(ocp.model.x, ocp.model.u)
        ocp.model.cost_y_expr = cs.vertcat(ocp.model.x, ocp.model.u)
        ocp.model.cost_y_expr_e = ocp.model.x

        ocp.solver_options = define_acados_ocp_options(config=config["ocp_options"])

        # Build cost function

        # ocp.solver_options = config.ocp_options

        ocp.code_export_directory = config["meta"]["code_export_dir"]

        self.ocp = ocp

        ocp_generate_external_functions(ocp, ocp.model)

        nlp, self.idx = build_nlp(ocp=self.ocp)

        # TODO: Move the constraints set to the corresponding MPC function. This function should only update the multipliers
        for stage in range(ocp.dims.N):
            # nlp.w.val["x", stage] = ocp_solver.get(stage, "x")
            # nlp.w.val["u", stage] = ocp_solver.get(stage, "u")

            if stage == 0:
                nlp.lbw.val["lbx", stage] = ocp.constraints.lbx_0
                nlp.ubw.val["ubx", stage] = ocp.constraints.ubx_0
            else:
                nlp.lbw.val["lbx", stage] = ocp.constraints.lbx
                nlp.ubw.val["ubx", stage] = ocp.constraints.ubx

            nlp.lbw.val["lbu", stage] = ocp.constraints.lbu
            nlp.ubw.val["ubu", stage] = ocp.constraints.ubu

        self.muliplier_map = LagrangeMultiplierMap(constraints=ocp.constraints, N=ocp.dims.N)

        nlp.L.sym = nlp.cost.sym + cs.dot(nlp.pi.sym, nlp.g.sym) + cs.dot(nlp.lam.sym, nlp.h.sym)
        nlp.L.fun = cs.Function(
            "L",
            [nlp.w.sym, nlp.lbw.sym, nlp.ubw.sym, nlp.pi.sym, nlp.lam.sym, nlp.p.sym, nlp.dT.sym],
            [nlp.L.sym],
            ["w", "lbw", "ubw", "pi", "lam", "p", "dT"],
            ["L"],
        )

        nlp.dL_dw.sym = cs.jacobian(nlp.L.sym, nlp.w.sym)

        arg_list, name_list = find_nlp_entry_expr_dependencies(nlp, "dL_dw", ["w", "lbw", "ubw", "pi", "lam", "p", "dT"])

        nlp.dL_dw.fun = cs.Function(
            "dL_dw",
            arg_list,
            [nlp.dL_dw.sym],
            name_list,
            ["dL_dw"],
        )

        # Check if nlp.dL_dw.sym is function of nlp.w.sym

        nlp.dL_dp.sym = cs.jacobian(nlp.L.sym, nlp.p.sym)

        arg_list, name_list = find_nlp_entry_expr_dependencies(nlp, "dL_dp", ["w", "lbw", "ubw", "pi", "lam", "p", "dT"])

        nlp.dL_dp.fun = cs.Function(
            "dL_dp",
            arg_list,
            [nlp.dL_dp.sym],
            name_list,
            ["dL_dp"],
        )

        nlp.R.sym = cs.vertcat(cs.transpose(nlp.dL_dw.sym), nlp.g.sym, nlp.lam.sym * nlp.h.sym)

        arg_list, name_list = find_nlp_entry_expr_dependencies(nlp, "R", ["w", "lbw", "ubw", "pi", "lam", "p", "dT"])

        nlp.R.fun = cs.Function(
            "R",
            arg_list,
            [nlp.R.sym],
            name_list,
            ["R"],
        )

        z = cs.vertcat(nlp.w.sym, nlp.pi.sym, nlp.lam.sym)

        # Generate sensitivity of the KKT matrix with respect to primal-dual variables
        nlp.dR_dz.sym = cs.jacobian(nlp.R.sym, z)
        arg_list, name_list = find_nlp_entry_expr_dependencies(nlp, "dR_dz", ["w", "lbw", "ubw", "pi", "lam", "p", "dT"])
        nlp.dR_dz.fun = cs.Function(
            "dR_dz",
            arg_list,
            [nlp.dR_dz.sym],
            name_list,
            ["dR_dz"],
        )

        nlp.dR_dp.sym = cs.jacobian(nlp.R.sym, nlp.p.sym)
        arg_list, name_list = find_nlp_entry_expr_dependencies(nlp, "dR_dp", ["w", "lbw", "ubw", "pi", "lam", "p", "dT"])
        nlp.dR_dp.fun = cs.Function(
            "dR_dp",
            arg_list,
            [nlp.dR_dp.sym],
            name_list,
            ["dR_dp"],
        )

        self.nlp = nlp

        # Check path to config.meta.json file. Create the directory if it does not exist.
        if not os.path.exists(os.path.dirname(config["meta"]["json_file"])):
            os.makedirs(os.path.dirname(config["meta"]["json_file"]))

        # TODO: Add config entries for json file and c_generated_code folder, and build, generate flags
        if build:
            self.ocp_solver = AcadosOcpSolver(ocp, json_file=config["meta"]["json_file"])
        else:
            # Assumes json file and c_generated_code folder already exists
            self.ocp_solver = AcadosOcpSolver(ocp, json_file=config["meta"]["json_file"], build=False, generate=False)

        self._parameters = ocp.parameter_values

    def reset(self, x0: np.ndarray):
        self.ocp_solver.reset()

        for stage in range(self.ocp_solver.acados_ocp.dims.N + 1):
            # self.ocp_solver.set(stage, "x", self.ocp_solver.acados_ocp.constraints.lbx_0)
            self.ocp_solver.set(stage, "x", x0)

    def set(self, stage, field, value):
        self.ocp_solver.set(stage, field, value)

        if field == "p":
            self.nlp.p.val = value

    def get(self, stage, field):
        return self.ocp_solver.get(stage, field)

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low = self.ocp.constraints.lbu
        high = self.ocp.constraints.ubu

        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low = self.ocp.constraints.lbu
        high = self.ocp.constraints.ubu

        return 0.5 * (high - low) * (action + 1.0) + low

    # def update(self) -> int:
    #     status = self.ocp_solver.solve()

    #     self.nlp = update_nlp(self.nlp, self.ocp_solver, self.muliplier_map)

    #     test_nlp_sanity(self.nlp)

    #     return status

    def q_update(self, x0: np.ndarray, u0: np.ndarray) -> int:
        """
        Update the solution of the OCP solver.

        Args:
            x0: Initial state.

        Returns:
            status: Status of the solver.
        """
        # Set initial state
        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)

        self.nlp.lbw.val["lbx", 0] = x0
        self.nlp.ubw.val["ubx", 0] = x0

        # Set initial action (needed for state-action value)
        self.ocp_solver.set(0, "u", u0)
        self.ocp_solver.set(0, "lbu", u0)
        self.ocp_solver.set(0, "ubu", u0)

        self.nlp.lbw.val["lbu", 0] = u0
        self.nlp.ubw.val["ubu", 0] = u0

        # Solve the optimization problem

        status = self.ocp_solver.solve()

        self.ocp_solver.set(0, "lbu", self.ocp_solver.acados_ocp.constraints.lbu)
        self.ocp_solver.set(0, "ubu", self.ocp_solver.acados_ocp.constraints.ubu)

        self.nlp = update_nlp(self.nlp, self.ocp_solver, self.muliplier_map)

        # test_nlp_sanity(self.nlp)

        return status

    def update(self, x0: np.ndarray) -> int:
        """
        Update the solution of the OCP solver.

        Args:
            x0: Initial state.

        Returns:
            status: Status of the solver.
        """
        # Set initial state
        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)

        self.nlp.lbw.val["lbx", 0] = x0
        self.nlp.ubw.val["ubx", 0] = x0

        # Set initial action (needed for state-action value)
        # u0 = np.zeros((self.ocp.dims.nu,))
        # self.ocp_solver.set(0, "u", u0)
        # self.ocp_solver.set(0, "lbu", u0)
        # self.ocp_solver.set(0, "ubu", u0)

        # Solve the optimization problem
        status = self.ocp_solver.solve()

        # self.nlp = update_nlp(self.nlp, self.ocp_solver, self.muliplier_map)
        self.update_nlp()

        # test_nlp_sanity(self.nlp)

        return status

    def update_nlp(self) -> None:
        """
        Update the NLP with the solution of the OCP solver.
        """
        self.nlp = update_nlp(self.nlp, self.ocp_solver, self.muliplier_map)

    def set_theta(self, theta: np.ndarray) -> None:
        """
        Set the value of the parameters.

        Args:
            theta: Parameters.
        """
        # self._parameters = theta
        # self.ocp_solver.set(0, "p", theta)
        # self.nlp.p.val = theta

        for stage in range(self.ocp_solver.acados_ocp.dims.N + 1):
            self.set(stage, "p", theta)

        self.nlp.p.val = theta

        print("hallo")

    def get_theta(self) -> np.ndarray:
        return self.nlp.p.val

    def get_parameters(self) -> np.ndarray:
        return self._parameters

    def get_dL_dp(self) -> np.ndarray:
        """
        Get the value of the sensitivity of the Lagrangian with respect to the parameters.

        Returns:
            dL_dp: Sensitivity of the Lagrangian with respect to the parameters.
        """
        return self.nlp.dL_dp.val.full().flatten()

    def get_L(self) -> float:
        """
        Get the value of the Lagrangian.

        Returns:
            L: Lagrangian.
        """
        return self.nlp.L.val

    def get_V(self) -> float:
        """
        Get the value of the value function.

        Assumes OCP is solved for state.

        Returns:
            V: Value function.
        """
        return self.ocp_solver.get_cost()

    def get_pi(self) -> np.ndarray:
        """
        Get the value of the policy.

        Assumes OCP is solved for state.
        """
        return self.ocp_solver.get(0, "u")

    def get_dpi_dp(self) -> np.ndarray:
        """
        Get the value of the sensitivity of the policy with respect to the parameters.

        Assumes OCP is solved for state and parameters.
        """
        # return self.nlp.dpi_dp.val.full().flatten()
        pass

    def get_dV_dp(self) -> float:
        """
        Get the value of the sensitivity of the value function with respect to the parameters.

        Assumes OCP is solved for state and parameters.

        Returns:
            dV_dp: Sensitivity of the value function with respect to the parameters.
        """
        return self.get_dL_dp()

    def get_Q(self) -> float:
        """
        Get the value of the state-action value function.

        Assumes OCP is solved for state and action.

        Returns:
            Q: State-action value function.
        """
        return self.ocp_solver.get_cost()

    def get_dQ_dp(self) -> float:
        """
        Get the value of the sensitivity of the state-action value function with respect to the parameters.

        Assumes OCP is solved for state, action and parameters.

        Returns:
            dQ_dp: Sensitivity of the state-action value function with respect to the parameters.
        """
        return self.get_dL_dp()

    def get_action(self, x0: np.ndarray) -> np.ndarray:
        """
        Update the solution of the OCP solver.

        Args:
            x0: Initial state.

        Returns:
            u: Optimal control action.
        """
        # Set initial state
        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)

        self.nlp.lbw.val["lbx", 0] = x0
        self.nlp.ubw.val["ubx", 0] = x0

        # Solve the optimization problem
        self.status = self.ocp_solver.solve()

        # Get solution
        action = self.ocp_solver.get(0, "u")

        # Scale to [-1, 1] for gym
        action = self.scale_action(action)

        return action

    def get_predicted_state_trajectory(self) -> np.ndarray:
        """
        Get the predicted state trajectory.

        Returns:
            x: Predicted state trajectory.
        """
        x = np.zeros((self.ocp.dims.N + 1, self.ocp.dims.nx))

        for i in range(self.ocp.dims.N + 1):
            x[i, :] = self.ocp_solver.get(i, "x")

        return x

    def get_predicted_control_trajectory(self) -> np.ndarray:
        """
        Get the predicted control trajectory.

        Returns:
            u: Predicted control trajectory.
        """
        u = np.zeros((self.ocp.dims.N, self.ocp.dims.nu))

        for i in range(self.ocp.dims.N):
            u[i, :] = self.ocp_solver.get(i, "u")

        return u

    def plot_prediction(self) -> None:
        """
        Plot the predicted trajectory.
        """

        x = self.get_predicted_state_trajectory()
        u = self.get_predicted_control_trajectory()

        _, ax = plt.subplots(self.ocp.dims.nx + self.ocp.dims.nu, 1, figsize=(10, 7))

        for i in range(self.ocp.dims.nx):
            ax[i].plot(x[:, i], "-o")
            ax[i].grid(True)
            ax[i].set_ylabel(f"x_{i}")

        # Make a stairs plot for u
        ax[self.ocp.dims.nx].step(np.arange(0, u.shape[0]), u.flatten(), where="post")
        ax[self.ocp.dims.nx].grid(True)
        ax[self.ocp.dims.nx].set_ylabel("u")

        # Draw bounds
        lbx = self.ocp.constraints.lbx
        ubx = self.ocp.constraints.ubx
        lbu = self.ocp.constraints.lbu
        ubu = self.ocp.constraints.ubu

        for i in range(self.ocp.dims.nx):
            ax[i].plot(np.ones_like(x[:, i]) * lbx[i], "--", color="grey")
            ax[i].plot(np.ones_like(x[:, i]) * ubx[i], "--", color="gray")

        ax[self.ocp.dims.nx].plot(np.ones_like(u) * lbu, "--", color="gray")
        ax[self.ocp.dims.nx].plot(np.ones_like(u) * ubu, "--", color="gray")

        plt.show()

    def print_header(self) -> None:
        """
        Print the header for the data table.
        """
        print("{:>8} {:>8} {:>8} {:>8} {:>8}".format("x", "x_dot", "theta", "theta_dot", "u"))

    def print_data(self, x: np.ndarray, u: np.ndarray) -> None:
        """
        Print the data table.

        Args:
            x: State.
            u: Control.
        """
        print("{:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f}".format(x[0], x[1], x[2], x[3], u))
