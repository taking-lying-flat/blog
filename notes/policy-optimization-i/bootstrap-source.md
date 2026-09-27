```python Bootstrap · 计算优势与价值目标 | OpenAI Spinning Up · PyTorch
def finish_path(self, last_val=0):
    """
    Call this at the end of a trajectory, or when one gets cut off
    by an epoch ending. This looks back in the buffer to where the
    trajectory started, and uses rewards and value estimates from
    the whole trajectory to compute advantage estimates with GAE-Lambda,
    as well as compute the rewards-to-go for each state, to use as
    the targets for the value function.

    The "last_val" argument should be 0 if the trajectory ended
    because the agent reached a terminal state (died), and otherwise
    should be V(s_T), the value function estimated for the last state.
    This allows us to bootstrap the reward-to-go calculation to account
    for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
    """

    path_slice = slice(self.path_start_idx, self.ptr)
    rews = np.append(self.rew_buf[path_slice], last_val)
    vals = np.append(self.val_buf[path_slice], last_val)

    # the next two lines implement GAE-Lambda advantage calculation
    deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
    self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

    # the next line computes rewards-to-go, to be targets for the value function
    self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

    self.path_start_idx = self.ptr
```
