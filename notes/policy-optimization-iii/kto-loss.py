def kto_loss(
    self, policy_chosen_logps: torch.FloatTensor, policy_rejected_logps: torch.FloatTensor,
    policy_KL_logps: torch.FloatTensor, reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor, reference_KL_logps: torch.FloatTensor,
) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    kl = (policy_KL_logps - reference_KL_logps).mean().detach()
    kl = self.accelerator.gather_for_metrics(kl).mean().clamp(min=0)

    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - kl))
    chosen_rewards = self.beta * chosen_logratios.detach()
    rejected_logratios = policy_rejected_logps - reference_rejected_logps
    rejected_losses = 1 - F.sigmoid(self.beta * (kl - rejected_logratios))
    rejected_rewards = self.beta * rejected_logratios.detach()

    losses = torch.cat(
        (self.desirable_weight * chosen_losses, self.undesirable_weight * rejected_losses), 0
    )
    return losses, chosen_rewards, rejected_rewards, kl
