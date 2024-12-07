import torch
from torch.distributions import Categorical
from torch.distributions.utils import logits_to_probs

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))


from actor_critic import ActorBase, create_network


"""

# Citation: we also referred to: https://github.com/abhaybd/Fleet-AI but made various modifications to fit our cause and to make it easier for us to understand. We had to understand the code and delete quite a bit but still make it work to play the battleship game. Initially we had looked at just the openai baselines library implementation but due to some architecture and hyperparameter issues it wasn't learning as well, so we found this other implementation and modified it

"""



class BattleshipActor(ActorBase):
    def __init__(self, device, state_dim, action_dim, layers=(256, 256)):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.logits_net = create_network((state_dim,) + layers + (action_dim,)).to(device)
        self.forward_probs = False

    def forward(self, state):
        if self.forward_probs:
            return self.probs(state)
        else:
            return self._distribution(state)

    def _distribution(self, state):
        output = self.logits_net(state)
        return Categorical(logits=output)

    def _log_prob(self, dist, actions):
        return dist.log_prob(actions.squeeze()).unsqueeze(-1)

    def greedy(self, state):
        return self._distribution(state).probs.argmax()

    def probs(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
        logits = self.logits_net(state)
        probs = logits_to_probs(logits)
        return probs
