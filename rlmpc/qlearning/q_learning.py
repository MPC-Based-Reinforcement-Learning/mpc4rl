import numpy as np

from replay_buffer import ReplayBuffer
from helpers import tqdm_context


class Q_learning:
    def __init__(self, ocp, agent_params, seed):
        # hyperparameters
        self.ocp = ocp
        self._parse_agent_params(**agent_params)

    def train(self, replay_buffer: ReplayBuffer):
        """
        Updates Qfn parameters by using (random) data from replay_buffer

        Parameters
        ----------
        replay_buffer : ReplayBuffer object
            Class instance containing all past data

        Returns
        -------
        dict : {l_theta: parameters of Qfn,
                TD_error: observed TD_error}

        """
        (_, _, actions, rewards, _, _, _, infos) = replay_buffer.flatten_buffer()
        td_avg, del_J = 0.0, 0.0
        n_steps = len(actions)
        for j in range(n_steps - 1):
            a, next_a = actions[j], actions[j + 1]
            q = infos[j]["soln"]["f"].full()[0, 0] - infos[j]["pf"][
                self.ocp.obs_dim
                + self.ocp.action_dim : self.ocp.obs_dim
                + 2 * self.ocp.action_dim,
                0,
            ].dot(a)
            q_next = infos[j + 1]["soln"]["f"].full()[0, 0] - infos[j + 1]["pf"][
                self.ocp.obs_dim
                + self.ocp.action_dim : self.ocp.obs_dim
                + 2 * self.ocp.action_dim,
                0,
            ].dot(next_a)
            td_error = rewards[j][0] + self.ocp.gamma * q_next - q
            td_avg += td_error
            grad_q = self.ocp.dVdP(
                infos[j]["soln"],
                infos[j]["pf"],
                infos[j]["p"],
                optimal=infos[j]["optimal"] and infos[j + 1]["optimal"],
            )
            del_J -= td_error * grad_q.T

        # Param update
        self.ocp.param_update(
            del_J / n_steps, constrained_updates=self.constrained_updates
        )
        # print info
        print(td_avg / n_steps)
        print(self.ocp.p_val.T)
        return {"theta": self.ocp.p_val.copy(), "TD_error": td_avg / n_steps}

    def _parse_agent_params(self, lr, tr, constrained_updates=False):
        self.lr = lr
        self.tr = tr
        self.constrained_updates = constrained_updates
