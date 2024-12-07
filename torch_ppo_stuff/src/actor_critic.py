import torch
from torch import nn
from torch.distributions import Categorical, Normal, Distribution
import numpy as np


# from the openai official library code: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/core.py
# we also referred to: https://github.com/abhaybd/Fleet-AI

class MultiCategorical(Distribution):
    def __init__(self, dists):
        super().__init__(validate_args=False)
        self.dists = dists

    def log_prob(self, value):
        ans = []
        for d, v in zip(self.dists, torch.split(value, 1, dim=-1)):
            ans.append(d.log_prob(v.squeeze(-1)))
        return torch.stack(ans, dim=-1).sum(dim=-1)

    def entropy(self):
        return torch.stack([d.entropy() for d in self.dists], dim=-1).sum(dim=-1)

    def sample(self, sample_shape=torch.Size()):
        return torch.stack([d.sample(sample_shape) for d in self.dists], dim=-1)

    def argmax(self):
        return torch.tensor([d.probs.argmax() for d in self.dists])
    


def create_network(layer_sizes, activation=nn.Tanh, end_activation=nn.Identity):
    layers = []
    for i in range(len(layer_sizes) - 1):
        act = activation if i < len(layer_sizes)-2 else end_activation
        layers += (nn.Linear(layer_sizes[i], layer_sizes[i + 1]), act())
    return nn.Sequential(*layers)


class ActorBase(nn.Module):

    def eval_actions(self, states, actions):
        dist = self._distribution(states)
        return dist.log_prob(actions).unsqueeze(-1), dist.entropy().mean()
    
    def greedy(self, state):
        return self._distribution(state).argmax()



class MultiDiscActor(ActorBase):
    def __init__(self, device, state_dim, action_dims, layers=(64,64)):
        super().__init__()
        self.device = device
        self.base_net = create_network((state_dim,) + layers, end_activation=nn.Tanh).to(device)
        self.outputs = nn.ModuleList()
        for n in action_dims:
            self.outputs.append(nn.Linear(layers[-1], n).to(device))

    def _distribution(self, state):
        base_out = self.base_net(state)
        dists = [Categorical(logits=layer(base_out)) for layer in self.outputs]
        return MultiCategorical(dists)

    def to(self, device):
        super().to(device)
        self.device = device
        return self



class Critic(nn.Module):
    def __init__(self, device, state_dim, layers=(64, 64)):
        super().__init__()

        self.v_net = create_network((state_dim,) + layers + (1,)).to(device)

    def forward(self, state):
        return self.v_net(state)
