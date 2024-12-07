import torch
import numpy as np



"""
main references from official openai baselines library
    https://github.com/openai/baselines/tree/master/baselines/ppo2
    https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/

# Citation: we also referred to: https://github.com/abhaybd/Fleet-AI but made various modifications to fit our cause and to make it easier for us to understand. We had to understand the code and delete quite a bit but still make it work to play the battleship game. Initially we had looked at just the openai baselines library implementation but due to some architecture and hyperparameter issues it wasn't learning as well, so we found this other implementation and modified it

"""


class PPOBuffer(object):
    def __init__(self):
        self.trajectories = []

    def create_traj(self):
        traj_id = len(self.trajectories)
        self.trajectories.append(_Traj())
        return traj_id

    def put_single_data(self, traj_id, state, action, log_prob, reward):
        if isinstance(state, (float, int)):
            state = np.array([state])
        if isinstance(action, (float, int)):
            action = np.array([action])
        traj = self.trajectories[traj_id]
    
        traj.states.append(state)
        traj.actions.append(action)
        traj.log_probs.append([log_prob])
        traj.rewards.append([reward])

    def put_data(self, data):
        for d in data:
            self.put_single_data(*d)

    def finish_traj(self, traj_id, last_val):
    
        self.trajectories[traj_id].last_val = last_val

    def get(self, device):

        to_tensor = lambda x: torch.from_numpy(np.array(x, dtype=np.float32)).to(device)

        states = [to_tensor(traj.states) for traj in self.trajectories]
        actions = [to_tensor(traj.actions) for traj in self.trajectories]
        log_probs = [to_tensor(traj.log_probs) for traj in self.trajectories]
        rewards = [to_tensor(traj.rewards) for traj in self.trajectories]
        last_vals = [to_tensor([traj.last_val]) for traj in self.trajectories]
        self.clear()
        return states, actions, log_probs, rewards, last_vals

    def clear(self):
        self.trajectories.clear()

    def size(self):
        return sum(map(len, self.trajectories))

class _Traj(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.last_val = None

    def __len__(self):
        return len(self.states)
