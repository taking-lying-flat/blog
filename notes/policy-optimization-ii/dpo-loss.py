def forward(
    self, policy_chosen_logps: torch.Tensor, policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor, reference_rejected_logps: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios

    if self.ipo:
        losses = (logits - 1 / (2 * self.beta)) ** 2
    else:
        losses = (-F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                  - F.logsigmoid(-self.beta * logits) * self.label_smoothing)

    loss = losses.mean()
    chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
    return loss, chosen_rewards, rejected_rewards
