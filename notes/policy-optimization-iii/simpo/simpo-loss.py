def simpo_loss(
    self, policy_chosen_logps: torch.FloatTensor, policy_rejected_logps: torch.FloatTensor,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    pi_logratios = pi_logratios.to(self.accelerator.device)
    logits = pi_logratios - self.gamma_beta_ratio
    losses = (
        -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
        - F.logsigmoid(-self.beta * logits) * self.label_smoothing
    )
    chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
    rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
    return losses, chosen_rewards, rejected_rewards
