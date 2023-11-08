from stable_baselines3.common.policies import BasePolicy, Schedule

# Imports from stable_baselines3.common. Can be removed later.
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
import torch as th
import torch.nn as nn

from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from rlmpc.common.mpc import MPC

from gymnasium import spaces


class MPCActorCriticPolicy(BasePolicy):
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

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """

        # # Preprocess the observation if needed
        # features = self.extract_features(obs)

        # if self.share_features_extractor:
        #     latent_pi, latent_vf = self.mlp_extractor(features)
        # else:
        #     pi_features, vf_features = features
        #     latent_pi = self.mlp_extractor.forward_actor(pi_features)
        #     latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # # Evaluate the values for the given observations

        # values = self.value_net(latent_vf)

        # distribution = self._get_action_dist_from_latent(latent_pi)

        # actions = distribution.get_actions(deterministic=deterministic)

        # log_prob = distribution.log_prob(actions)

        # actions = actions.reshape((-1, *self.action_space.shape))
        # return actions, values, log_prob

        # Throw error if called
        raise NotImplementedError("forward not implemented for MPCActorCriticPolicy")

    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
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
        raise NotImplementedError(
            "evaluate_actions not implemented for MPCActorCriticPolicy"
        )

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
        raise NotImplementedError(
            "predict_values not implemented for MPCActorCriticPolicy"
        )
