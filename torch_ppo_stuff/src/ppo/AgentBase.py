import torch

# Citation: we referred to: https://github.com/abhaybd/Fleet-AI but made various modifications to fit our cause and to make it easier for us to understand. We had to understand the code and delete quite a bit but still make it work to play the battleship game
class AgentBase(object):
    def __init__(self, *args):
        self.init_args = args
        self._total_it = torch.tensor([0], dtype=torch.int64)
        self._shared_memory = [self._total_it]

    @property
    def total_it(self):
        return self._total_it.item()

    @total_it.setter
    def total_it(self, value):
        self._total_it[0] = value

    def share_memory(self):
        for x in self._shared_memory:
            if hasattr(x, "share_memory"):
                x.share_memory()
            elif hasattr(x, "share_memory_"):
                x.share_memory_()

    def reset(self):
        pass

    def copy(self):
        constructor = type(self)
        other = constructor(*self.init_args)
        other.copy_from(self)
        return other

    def copy_from(self, other):
        self.total_it = other.total_it

    def _save_dict(self):
        return {"total_it": self.total_it}

    def _load_save_dict(self, save_dict):
        self.total_it = save_dict["total_it"]
