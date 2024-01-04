from torch.nn.parameter import Parameter
from stable_baselines3.common.policies import (
    BaseModel,
    BasePolicy,
    Schedule,
)

# Imports from stable_baselines3.common. Can be removed later.
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    NatureCNN,
    get_actor_critic_arch,
)
import torch as th
import torch.nn as nn

from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, TypeVar, Union

from rlmpc.common.mpc import MPC

from gymnasium import spaces

from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)

from stable_baselines3.td3.policies import (
    Actor,
    CnnPolicy,
    MlpPolicy,
    MultiInputPolicy,
    TD3Policy,
)

from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

from stable_baselines3.common.utils import (
    get_device,
    is_vectorized_observation,
    obs_as_tensor,
)


class ContinuousCritic(BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks: List[nn.Module] = []
        for idx in range(n_critics):
            q_net_list = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
            q_net = nn.Sequential(*q_net_list)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        return self.q_networks[0](th.cat([features, actions], dim=1))


class Actor(BasePolicy):
    """
    Actor MPC (policy) for TD3.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    mpc: MPC
    optimizer: th.optim.Optimizer

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        # net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        mpc: MPC,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn

        self.mpc = mpc

        # action_dim = get_action_dim(self.action_space)
        # actor_net = create_mlp(features_dim, action_dim, net_arch, activation_fn, squash_output=True)
        # Deterministic action
        # self.mu = nn.Sequential(*actor_net)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features = self.extract_features(obs, self.features_extractor)

        # TODO: Check features.size() == self.features_dim
        if features.flatten().size()[0] == self.features_dim:
            return self._predict(features)
        else:  # Batch of observations
            # print("Features: ", features[:10, :])
            # for feature in features[:10, :]:
            #     print("Feature: ", feature)
            return th.stack([self._predict(observation) for observation in obs])

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        # Assume that the observation is the state.

        # TODO: Same as in ppo/policies.py. Refactor.

        # Convert observation to numpy array
        observation = observation.cpu().numpy().reshape(-1)

        # Get the action from the MPC
        action = self.mpc.get_action(observation)

        # Convert action to tensor
        action = th.tensor(action, dtype=th.float32)

        return action

    def parameters(self) -> th.Tensor:
        # Get the parameters from the MPC
        parameters = self.mpc.get_parameters()

        # Convert parameters from np.ndarray to an Iterator of th.Tensor
        parameters = (th.tensor(parameter, dtype=th.float32) for parameter in parameters)

        return parameters


class MPCTD3Policy(BasePolicy):
    """
    Policy class (with both actor and critic) for TD3.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    actor: Actor
    actor_target: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    # device: th.device

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        mpc: MPC,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 1,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )

        # Default network architecture, from the original paper
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = [256, 256]
            else:
                net_arch = [400, 300]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }

        self.actor_kwargs = {
            "mpc": mpc,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
        }

        self.critic_kwargs = self.net_args.copy()

        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        # Create actor and target
        # the features extractor should not be shared
        self.actor = self.make_actor(features_extractor=None)
        self.actor_target = self.make_actor(features_extractor=None)

        # TODO: Implement this for MPC-based Actor.
        # Initialize the target to have the same weights as the actor
        # self.actor_target.load_state_dict(self.actor.state_dict())

        # NOTE: The actor optimizer is not being used.
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Critic target should not share the features extractor with critic
            # but it can share it with the actor target as actor and critic are sharing
            # the same features_extractor too
            # NOTE: as a result the effective poliak (soft-copy) coefficient for the features extractor
            # will be 2 * tau instead of tau (updated one time with the actor, a second time with the critic)
            self.critic_target = self.make_critic(features_extractor=self.actor_target.features_extractor)
        else:
            # Create new features extractor for each network
            self.critic = self.make_critic(features_extractor=None)
            self.critic_target = self.make_critic(features_extractor=None)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.optimizer_class(
            self.critic.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        # Target networks should always be in eval mode
        self.actor_target.set_training_mode(False)
        self.critic_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                share_features_extractor=self.share_features_extractor,
            )
        )
        return data

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self._predict(observation, deterministic=deterministic)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self.actor(observation)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode


# class MPCTD3Policy(BasePolicy):
#     """
#     Policy class (with both MPC actor as and NN critic) for TD3.

#     :param observation_space: Observation space
#     :param action_space: Action space
#     :param lr_schedule: Learning rate schedule (could be constant)
#     :param net_arch: The specification of the policy and value networks.
#     :param activation_fn: Activation function
#     :param features_extractor_class: Features extractor to use.
#     :param features_extractor_kwargs: Keyword arguments
#         to pass to the features extractor.
#     :param normalize_images: Whether to normalize images or not,
#          dividing by 255.0 (True by default)
#     :param optimizer_class: The optimizer to use,
#         ``th.optim.Adam`` by default
#     :param optimizer_kwargs: Additional keyword arguments,
#         excluding the learning rate, to pass to the optimizer
#     :param n_critics: Number of critic networks to create.
#     :param share_features_extractor: Whether to share or not the features extractor
#         between the actor and the critic (this saves computation time)
#     """

#     actor: Actor
#     actor_target: Actor
#     critic: ContinuousCritic
#     critic_target: ContinuousCritic

#     def __init__(
#         self,
#         mpc: MPC,
#         observation_space: spaces.Space,
#         action_space: spaces.Box,
#         lr_schedule: Schedule,
#         net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
#         activation_fn: Type[nn.Module] = nn.ReLU,
#         features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
#         features_extractor_kwargs: Optional[Dict[str, Any]] = None,
#         normalize_images: bool = True,
#         optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
#         optimizer_kwargs: Optional[Dict[str, Any]] = None,
#         n_critics: int = 2,
#         share_features_extractor: bool = False,
#     ):
#         super().__init__(
#             observation_space,
#             action_space,
#             features_extractor_class,
#             features_extractor_kwargs,
#             optimizer_class=optimizer_class,
#             optimizer_kwargs=optimizer_kwargs,
#             squash_output=True,
#             normalize_images=normalize_images,
#         )

#         self.mpc = MPC

#         # Default network architecture, from the original paper
#         if net_arch is None:
#             if features_extractor_class == NatureCNN:
#                 net_arch = [256, 256]
#             else:
#                 net_arch = [400, 300]

#         actor_arch, critic_arch = get_actor_critic_arch(net_arch)

#         self.net_arch = net_arch
#         self.activation_fn = activation_fn
#         self.net_args = {
#             "observation_space": self.observation_space,
#             "action_space": self.action_space,
#             "net_arch": actor_arch,
#             "activation_fn": self.activation_fn,
#             "normalize_images": normalize_images,
#         }

#         self.actor_kwargs = {
#             "mpc": mpc,
#             "observation_space": self.observation_space,
#             "action_space": self.action_space,
#         }

#         self.critic_kwargs = self.net_args.copy()
#         self.critic_kwargs.update(
#             {
#                 "n_critics": n_critics,
#                 "net_arch": critic_arch,
#                 "share_features_extractor": share_features_extractor,
#             }
#         )

#         self.share_features_extractor = share_features_extractor

#         self._build(lr_schedule)

#     def _build(self, lr_schedule: Schedule) -> None:
#         # Create actor and target
#         # the features extractor should not be shared
#         self.actor = self.make_actor(features_extractor=None)

#         self.actor_target = self.make_actor(features_extractor=None)
#         # Initialize the target to have the same weights as the actor
#         self.actor_target.load_state_dict(self.actor.state_dict())

#         # self.actor.optimizer = self.optimizer_class(
#         #     self.actor.parameters(),
#         #     lr=lr_schedule(1),  # type: ignore[call-arg]
#         #     **self.optimizer_kwargs,
#         # )

#         if self.share_features_extractor:
#             self.critic = self.make_critic(
#                 features_extractor=self.actor.features_extractor
#             )
#             # Critic target should not share the features extractor with critic
#             # but it can share it with the actor target as actor and critic are sharing
#             # the same features_extractor too
#             # NOTE: as a result the effective poliak (soft-copy) coefficient for the features extractor
#             # will be 2 * tau instead of tau (updated one time with the actor, a second time with the critic)
#             self.critic_target = self.make_critic(
#                 features_extractor=self.actor_target.features_extractor
#             )
#         else:
#             # Create new features extractor for each network
#             self.critic = self.make_critic(features_extractor=None)
#             self.critic_target = self.make_critic(features_extractor=None)

#         self.critic_target.load_state_dict(self.critic.state_dict())
#         self.critic.optimizer = self.optimizer_class(
#             self.critic.parameters(),
#             lr=lr_schedule(1),  # type: ignore[call-arg]
#             **self.optimizer_kwargs,
#         )

#         # Target networks should always be in eval mode
#         self.actor_target.set_training_mode(False)
#         self.critic_target.set_training_mode(False)

#     def _get_constructor_parameters(self) -> Dict[str, Any]:
#         data = super()._get_constructor_parameters()

#         data.update(
#             dict(
#                 net_arch=self.net_arch,
#                 activation_fn=self.net_args["activation_fn"],
#                 n_critics=self.critic_kwargs["n_critics"],
#                 lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
#                 optimizer_class=self.optimizer_class,
#                 optimizer_kwargs=self.optimizer_kwargs,
#                 features_extractor_class=self.features_extractor_class,
#                 features_extractor_kwargs=self.features_extractor_kwargs,
#                 share_features_extractor=self.share_features_extractor,
#             )
#         )
#         return data

#     def make_actor(
#         self, features_extractor: Optional[BaseFeaturesExtractor] = None
#     ) -> Actor:
#         actor_kwargs = self._update_features_extractor(
#             self.actor_kwargs, features_extractor
#         )
#         return Actor(**actor_kwargs).to(self.device)

#     def make_critic(
#         self, features_extractor: Optional[BaseFeaturesExtractor] = None
#     ) -> ContinuousCritic:
#         critic_kwargs = self._update_features_extractor(
#             self.critic_kwargs, features_extractor
#         )
#         return ContinuousCritic(**critic_kwargs).to(self.device)

#     def forward(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
#         return self._predict(observation, deterministic=deterministic)

#     def _predict(
#         self, observation: th.Tensor, deterministic: bool = False
#     ) -> th.Tensor:
#         # Note: the deterministic deterministic parameter is ignored in the case of TD3.
#         #   Predictions are always deterministic.
#         return self.actor(observation)

#     def set_training_mode(self, mode: bool) -> None:
#         """
#         Put the policy in either training or evaluation mode.

#         This affects certain modules, such as batch normalisation and dropout.

#         :param mode: if true, set to training mode, else set to evaluation mode
#         """
#         self.actor.set_training_mode(mode)
#         self.critic.set_training_mode(mode)
#         self.training = mode
