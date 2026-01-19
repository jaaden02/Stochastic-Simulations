from .setup import Grid, InitialCondition, DiffusionConfig, VelocitiesConfig
from .boundary_conditions import BoundaryConditions
from .logging_utils import configure_logging, get_logger
from . import plotting

__all__ = [
    'Grid',
    'InitialCondition',
    'DiffusionConfig',
    'VelocitiesConfig',
    'BoundaryConditions',
    'configure_logging',
    'get_logger',
    'plotting',
]

def hello() -> str:
    return "Hello from stochastic-simulations!"
