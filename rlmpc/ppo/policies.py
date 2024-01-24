from external.stable_baselines3.stable_baselines3.common.policies import (
    ActorCriticPolicy,
    BasePolicy,
    Schedule,
)

from external.stable_baselines3.stable_baselines3.ppo.policies import (
    MultiInputActorCriticPolicy,
)

# Imports from stable_baselines3.common. Can be removed later.
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
import torch as th
import torch.nn as nn

from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from mpc.common.mpc import MPC

from gymnasium import spaces


class MPCActorCriticPolicy(ActorCriticPolicy):
    mpc: MPC
    optimizer: th.optim.Optimizer

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        mpc: MPC,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )

        self.mpc = mpc

        print("MPCActorCriticPolicy initialized")

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        # Assume that the observation is the state.

        # Convert observation to numpy array
        observation = observation.cpu().numpy().reshape(-1)

        # Get the action from the MPC
        action = self.mpc.get_action(observation)

        # Convert action to tensor
        action = th.tensor(action, dtype=th.float32)

        return action

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        raise NotImplementedError("forward not implemented for MPCActorCriticPolicy")

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        # features = self.extract_features(obs)
        # if self.share_features_extractor:
        #     latent_pi, latent_vf = self.mlp_extractor(features)
        # else:
        #     pi_features, vf_features = features
        #     latent_pi = self.mlp_extractor.forward_actor(pi_features)
        #     latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # distribution = self._get_action_dist_from_latent(latent_pi)
        # log_prob = distribution.log_prob(actions)
        # values = self.value_net(latent_vf)
        # entropy = distribution.entropy()
        # return values, log_prob, entropy

        # Throw error if called
        raise NotImplementedError("evaluate_actions not implemented for MPCActorCriticPolicy")

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        # features = super().extract_features(obs, self.vf_features_extractor)
        # latent_vf = self.mlp_extractor.forward_critic(features)
        # return self.value_net(latent_vf)

        # Throw error if called
        raise NotImplementedError("predict_values not implemented for MPCActorCriticPolicy")


class MPCMultiInputActorCriticPolicy(MultiInputActorCriticPolicy):
    """
    MultiInputActorClass policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space (Tuple)
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Uses the CombinedExtractor
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    mpc: MPC

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        mpc: MPC,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

        self.mpc = mpc

        print("MPCActorCriticPolicy initialized")

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        # Assume that the observation is the state.

        # Convert observation to numpy array
        observation = observation.cpu().numpy().reshape(-1)

        # Get the action from the MPC
        action = self.mpc.get_action(observation)

        # Convert action to tensor
        action = th.tensor(action, dtype=th.float32)

        return action

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        raise NotImplementedError("forward not implemented for MPCActorCriticPolicy")

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        # features = super().extract_features(obs, self.vf_features_extractor)
        # latent_vf = self.mlp_extractor.forward_critic(features)
        # return self.value_net(latent_vf)

        # Throw error if called
        raise NotImplementedError("predict_values not implemented for MPCActorCriticPolicy")


# class ContinuousCritic(BaseModel):
#     """
#     Critic network(s) for DDPG/SAC/TD3.
#     It represents the action-state value function (Q-value function).
#     Compared to A2C/PPO critics, this one represents the Q-value
#     and takes the continuous action as input. It is concatenated with the state
#     and then fed to the network which outputs a single value: Q(s, a).
#     For more recent algorithms like SAC/TD3, multiple networks
#     are created to give different estimates.

#     By default, it creates two critic networks used to reduce overestimation
#     thanks to clipped Q-learning (cf TD3 paper).

#     :param observation_space: Obervation space
#     :param action_space: Action space
#     :param net_arch: Network architecture
#     :param features_extractor: Network to extract features
#         (a CNN when using images, a nn.Flatten() layer otherwise)
#     :param features_dim: Number of features
#     :param activation_fn: Activation function
#     :param normalize_images: Whether to normalize images or not,
#          dividing by 255.0 (True by default)
#     :param n_critics: Number of critic networks to create.
#     :param share_features_extractor: Whether the features extractor is shared or not
#         between the actor and the critic (this saves computation time)
#     """

#     features_extractor: BaseFeaturesExtractor

#     def __init__(
#         self,
#         observation_space: spaces.Space,
#         action_space: spaces.Box,
#         net_arch: List[int],
#         features_extractor: BaseFeaturesExtractor,
#         features_dim: int,
#         activation_fn: Type[nn.Module] = nn.ReLU,
#         normalize_images: bool = True,
#         n_critics: int = 2,
#         share_features_extractor: bool = True,
#     ):
#         super().__init__(
#             observation_space,
#             action_space,
#             features_extractor=features_extractor,
#             normalize_images=normalize_images,
#         )

#         action_dim = get_action_dim(self.action_space)

#         self.share_features_extractor = share_features_extractor
#         self.n_critics = n_critics
#         self.q_networks: List[nn.Module] = []
#         for idx in range(n_critics):
#             q_net_list = create_mlp(
#                 features_dim + action_dim, 1, net_arch, activation_fn
#             )
#             q_net = nn.Sequential(*q_net_list)
#             self.add_module(f"qf{idx}", q_net)
#             self.q_networks.append(q_net)

#     def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
#         # Learn the features extractor using the policy loss only
#         # when the features_extractor is shared with the actor
#         with th.set_grad_enabled(not self.share_features_extractor):
#             features = self.extract_features(obs, self.features_extractor)
#         qvalue_input = th.cat([features, actions], dim=1)
#         return tuple(q_net(qvalue_input) for q_net in self.q_networks)

#     def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
#         """
#         Only predict the Q-value using the first network.
#         This allows to reduce computation when all the estimates are not needed
#         (e.g. when updating the policy in TD3).
#         """
#         with th.no_grad():
#             features = self.extract_features(obs, self.features_extractor)
#         return self.q_networks[0](th.cat([features, actions], dim=1))
