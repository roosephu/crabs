import abc


class BaseDynModel(abc.ABC):
    @abc.abstractmethod
    def get_steps(self, states, actions):
        pass

    def get_init_states(self, n_envs=1):
        return None
