"""JAX simulator and RL environment for a belt-driven (CoreXY-style) gantry."""

from .baselines import make_pd_policy
from .dynamics import belt_quantities, energy, friction, rest_state, vector_field
from .env import EnvState, GantryEnv
from .integrator import step_zoh
from .params import EnvConfig, GantryParams, load_config
from .tasks import SETPOINT, TRAJECTORY

__all__ = [
    "EnvConfig", "EnvState", "GantryEnv", "GantryParams", "SETPOINT", "TRAJECTORY",
    "belt_quantities", "energy", "friction", "load_config", "make_pd_policy",
    "rest_state", "step_zoh", "vector_field",
]
