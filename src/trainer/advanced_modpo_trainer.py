from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

from src.trainer.modpo_trainer import MODPOTrainer
from src.utils.reward import RewardWrapperList, RewardWrapperInput


class AdvancedMODPOTrainer(MODPOTrainer):
    """Extended MODPO trainer with multiple scalarization methods."""

    def __init__(self, *args, scalarization_method="linear", **kwargs):
        super().__init__(*args, **kwargs)
        self.scalarization_method = scalarization_method

        # Parameters for different utility functions
        self.risk_aversion = kwargs.get('risk_aversion',
                                        1.0)  # For exponential utility
        self.risk_parameter = kwargs.get('risk_parameter',
                                         0.5)  # For power utility
        self.reference_point = kwargs.get('reference_point',
                                          None)  # For Chebyshev

    def modpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        chosen_margin_reward: torch.FloatTensor,
        rejected_margin_reward: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the MODPO loss with different scalarization methods."""

        if self.scalarization_method == "linear":
            return self._linear_scalarization(policy_chosen_logps,
                                              policy_rejected_logps,
                                              reference_chosen_logps,
                                              reference_rejected_logps,
                                              chosen_margin_reward,
                                              rejected_margin_reward)
        elif self.scalarization_method == "chebyshev":
            return self._chebyshev_scalarization(policy_chosen_logps,
                                                 policy_rejected_logps,
                                                 reference_chosen_logps,
                                                 reference_rejected_logps,
                                                 chosen_margin_reward,
                                                 rejected_margin_reward)
        elif self.scalarization_method == "exponential":
            return self._exponential_utility_scalarization(
                policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps,
                chosen_margin_reward, rejected_margin_reward)
        elif self.scalarization_method == "power":
            return self._power_utility_scalarization(policy_chosen_logps,
                                                     policy_rejected_logps,
                                                     reference_chosen_logps,
                                                     reference_rejected_logps,
                                                     chosen_margin_reward,
                                                     rejected_margin_reward)
        else:
            raise ValueError(
                f"Unknown scalarization method: {self.scalarization_method}")

    def _linear_scalarization(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        chosen_margin_reward: torch.FloatTensor,
        rejected_margin_reward: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Original MODPO implementation with linear scalarization."""
        return super().modpo_loss(policy_chosen_logps, policy_rejected_logps,
                                  reference_chosen_logps,
                                  reference_rejected_logps,
                                  chosen_margin_reward, rejected_margin_reward)

    def _chebyshev_scalarization(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        chosen_margin_reward: torch.FloatTensor,
        rejected_margin_reward: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Chebyshev scalarization that minimizes the maximum regret."""
        # Compute rewards for each objective
        chosen_rewards_kl = self.beta * (policy_chosen_logps -
                                         reference_chosen_logps)
        rejected_rewards_kl = self.beta * (policy_rejected_logps -
                                           reference_rejected_logps)

        # Stack rewards for each objective
        chosen_rewards = torch.stack(
            [chosen_rewards_kl, -chosen_margin_reward @ self.w[1:]], dim=-1)
        rejected_rewards = torch.stack(
            [rejected_rewards_kl, -rejected_margin_reward @ self.w[1:]],
            dim=-1)

        # Normalize by weights
        chosen_rewards = chosen_rewards / self.w.unsqueeze(0)
        rejected_rewards = rejected_rewards / self.w.unsqueeze(0)

        # Compute the maximum difference (Chebyshev distance)
        logits = (chosen_rewards - rejected_rewards).min(dim=-1)[0]

        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(logits)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - logits)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return losses, chosen_rewards_kl.detach(), rejected_rewards_kl.detach()

    def _exponential_utility_scalarization(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        chosen_margin_reward: torch.FloatTensor,
        rejected_margin_reward: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Exponential utility scalarization for risk-averse optimization."""
        # Compute base rewards
        chosen_rewards = (1 / self.w[0]) * (
            self.beta * (policy_chosen_logps - reference_chosen_logps) -
            chosen_margin_reward @ self.w[1:])
        rejected_rewards = (1 / self.w[0]) * (
            self.beta * (policy_rejected_logps - reference_rejected_logps) -
            rejected_margin_reward @ self.w[1:])

        # Apply exponential utility transformation
        chosen_utility = -torch.exp(-self.risk_aversion * chosen_rewards)
        rejected_utility = -torch.exp(-self.risk_aversion * rejected_rewards)

        logits = chosen_utility - rejected_utility

        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(logits)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - logits)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return losses, chosen_rewards.detach(), rejected_rewards.detach()

    def _power_utility_scalarization(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        chosen_margin_reward: torch.FloatTensor,
        rejected_margin_reward: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Power utility scalarization with adjustable risk sensitivity."""
        # Compute base rewards
        chosen_rewards = (1 / self.w[0]) * (
            self.beta * (policy_chosen_logps - reference_chosen_logps) -
            chosen_margin_reward @ self.w[1:])
        rejected_rewards = (1 / self.w[0]) * (
            self.beta * (policy_rejected_logps - reference_rejected_logps) -
            rejected_margin_reward @ self.w[1:])

        # Apply power utility transformation
        # Add a small positive constant to ensure rewards are positive
        eps = 1e-6
        chosen_utility = torch.sign(self.risk_parameter) * (
            chosen_rewards + eps).pow(self.risk_parameter)
        rejected_utility = torch.sign(self.risk_parameter) * (
            rejected_rewards + eps).pow(self.risk_parameter)

        logits = chosen_utility - rejected_utility

        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(logits)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - logits)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return losses, chosen_rewards.detach(), rejected_rewards.detach()
