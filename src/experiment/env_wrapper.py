import gymnasium as gym
import numpy as np


class RandomizeCartPolePhysics(gym.Wrapper):
    """
    Wrapper to slightly randomize the CartPole physical parameters.

    The random variation is applied once upon initialization based on the
    provided seed, ensuring consistent physics for a given worker or evaluation
    run across multiple episodes.
    """

    def __init__(self, env, seed=None, variance=0.2):
        super().__init__(env)
        self.variance = variance

        # Save nominal internal parameters
        self._nominal_gravity = env.unwrapped.gravity
        self._nominal_masscart = env.unwrapped.masscart
        self._nominal_masspole = env.unwrapped.masspole
        self._nominal_length = env.unwrapped.length
        self._nominal_force_mag = env.unwrapped.force_mag

        # Apply randomization once
        rng = np.random.RandomState(seed if seed is not None else np.random.randint(2**31))
        v = self.variance

        def rand_factor():
            return 1.0 + rng.uniform(-v, v)

        env_unwrapped = self.unwrapped
        env_unwrapped.gravity = self._nominal_gravity * rand_factor()
        env_unwrapped.masscart = self._nominal_masscart * rand_factor()
        env_unwrapped.masspole = self._nominal_masspole * rand_factor()
        env_unwrapped.length = self._nominal_length * rand_factor()
        env_unwrapped.force_mag = self._nominal_force_mag * rand_factor()

        # Recalculate derived parameters
        env_unwrapped.total_mass = env_unwrapped.masscart + env_unwrapped.masspole
        env_unwrapped.polemass_length = env_unwrapped.masspole * env_unwrapped.length

    def reset(self, *, seed=None, options=None):
        return super().reset(seed=seed, options=options)
