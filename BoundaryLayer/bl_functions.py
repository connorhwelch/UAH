import numpy as np

from dataclasses import dataclass, field
import numpy as np


@dataclass
class PerturbationVariable:
    observations: list[float] = field(default_factory=list)

    def add_observation(self, value: float | list[float] | np.ndarray):
        """Adds an instantaneous observation or a list/array of observations."""
        if isinstance(value, (list, np.ndarray)):
            self.observations.extend(value)
        else:
            self.observations.append(value)

    @property
    def mean(self) -> float:
        """Computes the Reynolds-averaged mean value."""
        return np.mean(self.observations) if self.observations else 0.0

    @property
    def perturbations(self) -> np.ndarray:
        """Computes perturbation values as deviations from the mean."""
        mean_value = self.mean
        return np.array([obs - mean_value for obs in self.observations], dtype=np.float64)

    @property
    def biased_std(self) -> float:
        """Computes the biased standard deviation (N denominator)."""
        return np.std(self.observations, ddof=0) if self.observations else 0.0

    def clear_observations(self):
        """Clears all stored observations."""
        self.observations.clear()


