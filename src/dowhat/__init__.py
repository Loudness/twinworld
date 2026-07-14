"""dowhat — counterfactual reasoning over symbolic state-transition traces."""

from .api import (
    DEFAULT_ABSTRACTIONS,
    Backtracking,
    CausalRepresentation,
    CounterfactualSet,
    IdentificationError,
    IdentifiedQuery,
    Interventional,
    Representational,
    compute,
    identify,
    model,
    refute,
)
from .engine import Counterfactual, Solution, Task, Trace, UnsolvedTaskError, solve
from .mechanisms import Identity, Mechanism, Recolor, Translate, candidate_primitives
from .metrics import MetricVector, evaluate, proximity, sparsity, validity
from .refute import RefutationReport
from .representation import Grid, Obj, StateGraph, as_grid, match_objects, parse_grid

__version__ = "0.0.1"

__all__ = [
    "Backtracking",
    "CausalRepresentation",
    "Counterfactual",
    "CounterfactualSet",
    "DEFAULT_ABSTRACTIONS",
    "Grid",
    "Representational",
    "IdentificationError",
    "IdentifiedQuery",
    "Identity",
    "Interventional",
    "Mechanism",
    "MetricVector",
    "Obj",
    "Recolor",
    "RefutationReport",
    "Solution",
    "StateGraph",
    "Task",
    "Trace",
    "Translate",
    "UnsolvedTaskError",
    "as_grid",
    "candidate_primitives",
    "compute",
    "evaluate",
    "identify",
    "match_objects",
    "model",
    "parse_grid",
    "proximity",
    "refute",
    "solve",
    "sparsity",
    "validity",
]
