import numpy as np
import src.environment.pendulum.export_model as pendulum_model
from acados_template import AcadosSim, AcadosSimSolver, AcadosSimDims
from matplotlib import pyplot as plt

def get_model(name: str, param: dict):
    if name == 'pendulum':
        return pendulum_model.export_pendulum_ode_model(param)
    else:
        raise NotImplementedError

class Environment(object):
    """docstring for Simulator."""
    def __init__(self, name: str, param: dict, acados_settings: dict, 
                 simulation_settings: dict, **kwargs):
        super(Environment, self).__init__()

        self.name = name
        self.param = param


        self.sim = AcadosSim(acados_settings['installation_directory'])
        self.sim.model = get_model(name, param)

        # set simulation time
        self.sim_opts = self.sim.solver_options
        self.sim.solver_options.T = simulation_settings['dt']

        # set options
        self.sim.solver_options.num_stages = 7
        self.sim.solver_options.num_steps = 3
        # self.sim.solver_options.newton_iter = 10 # for implicit integrator
        # self.sim.solver_options.collocation_type = "GAUSS_RADAU_IIA"
        self.sim.solver_options.integrator_type = "ERK"
        self.sim.solver_options.sens_forw = True
        self.sim.solver_options.sens_adj = True
        self.sim.solver_options.sens_hess = False
        self.sim.solver_options.sens_algebraic = False
        self.sim.solver_options.output_z = False
        self.sim.solver_options.sim_method_jac_reuse = False

        self.integrator = AcadosSimSolver(self.sim)

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        self.integrator.set("x", x)
        self.integrator.set("u", u)
        status = self.integrator.solve()

        if status != 0:
            raise Exception('acados acados_sim_solve() returned status {}. Exiting.'.format(status))
        
        return self.integrator.get('x')


    def simulate(self, x0: np.ndarray, u: np.ndarray, sim_param: dict) -> np.ndarray:

        t_sim = np.arange(sim_param['t0'], sim_param['T'], sim_param['dt'])
        x_sim = np.zeros((t_sim.shape[0], self.sim.dims.nx))
        u_sim = u

        x_sim[0, :] = x0

        for k, t_k in enumerate(t_sim[:-1]):
            self.integrator.set("x", x_sim[k, :])
            self.integrator.set("u", u_sim[k, :])
            status = self.integrator.solve()

            if status == 0:
                x_sim[k+1, :] = self.integrator.get('x')

        return t_sim, x_sim, u


    def reset(self):
        self.set_state(self.x0)

    def get_status(self):
        raise NotImplementedError

    def set_state(self, x: np.ndarray):
        self.x = x

    def set_initial_state(self, x0: np.ndarray):
        self.x0 = x0
    
    def get_state(self):
        raise NotImplementedError

    def get_state_dim(self):
        raise NotImplementedError
    
    def get_action_dim(self):
        raise NotImplementedError
    
    def get_state_bounds(self):
        raise NotImplementedError
    
    def get_action_bounds(self):
        raise NotImplementedError
    
    def get_state_labels(self):
        raise NotImplementedError
    
    def get_action_labels(self):
        raise NotImplementedError
    
    def get_state_units(self):
        raise NotImplementedError
    
    def get_action_units(self):
        raise NotImplementedError
    