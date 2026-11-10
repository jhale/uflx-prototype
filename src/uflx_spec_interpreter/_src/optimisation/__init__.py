"""Optional algebraic optimisation; importing this module does not load egglog."""

from .optimise import OptimisationOptions, OptimisationReport, OptimisationResult, optimise

__all__ = ["OptimisationOptions", "OptimisationReport", "OptimisationResult", "optimise"]
