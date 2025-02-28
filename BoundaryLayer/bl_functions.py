import numpy as np
import scipy
from dataclasses import dataclass, field
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.stats import linregress
import matplotlib.pyplot as plt


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


import numpy as np


class RichardsonNumber:
    def __init__(self, theta_i, theta_f, height_i, height_f, uwind_i, uwind_f, vwind_i, vwind_f, u_star=None,
                 theta_star=None):
        """
        Initialize the Richardson Number calculator.

        Parameters:
        - theta_i, theta_f: Potential temperature (K) at two heights.
        - height_i, height_f: Heights (m) of the two levels.
        - uwind_i, uwind_f: Zonal wind speed (m/s) at two heights.
        - vwind_i, vwind_f: Meridional wind speed (m/s) at two heights.
        - u_star: Friction velocity (m/s) for flux Richardson number (optional).
        - theta_star: Surface temperature scale (K) for flux Richardson number (optional).
        """
        self.theta_i = theta_i
        self.theta_f = theta_f
        self.height_i = height_i
        self.height_f = height_f
        self.uwind_i = uwind_i
        self.uwind_f = uwind_f
        self.vwind_i = vwind_i
        self.vwind_f = vwind_f
        self.u_star = u_star
        self.theta_star = theta_star

        # Constants
        self.g = 9.81  # Gravity (m/s²)

    def gradient_richardson(self):
        """Compute the Gradient Richardson Number (Ri_g)."""
        delta_theta = self.theta_f - self.theta_i
        delta_z = self.height_f - self.height_i
        delta_u = self.uwind_f - self.uwind_i
        delta_v = self.vwind_f - self.vwind_i

        if delta_z == 0 or (delta_u == 0 and delta_v == 0):
            return float("inf")  # Avoid division by zero

        shear = (delta_u ** 2 + delta_v ** 2) / delta_z ** 2
        buoyancy = (self.g / ((self.theta_i + self.theta_f) / 2)) * (delta_theta / delta_z)

        return buoyancy / shear if shear != 0 else float("inf")

    def bulk_richardson(self):
        """Compute the Bulk Richardson Number (Ri_b)."""
        delta_theta = self.theta_f - self.theta_i
        delta_z = self.height_f - self.height_i
        delta_u = self.uwind_f - self.uwind_i
        delta_v = self.vwind_f - self.vwind_i

        velocity_squared = delta_u ** 2 + delta_v ** 2
        if delta_z == 0 or velocity_squared == 0:
            return "ERROR: Undefined"  # Avoid division by zero

        buoyancy = (self.g / self.theta_i) * delta_theta * delta_z
        return buoyancy / velocity_squared

    def flux_richardson(self):
        """Compute the Flux Richardson Number (Ri_f) if u_star and theta_star are provided."""
        if self.u_star is None or self.theta_star is None or self.u_star == 0:
            return None  # Flux Richardson Number requires u_star and theta_star

        return (self.g / self.theta_i) * (self.theta_star / self.u_star ** 2)

    def compute_all(self):
        """Compute all Richardson Number variants."""
        return {
            "Gradient Richardson Number (Ri_g)": self.gradient_richardson(),
            "Bulk Richardson Number (Ri_b)": self.bulk_richardson(),
            "Flux Richardson Number (Ri_f)": self.flux_richardson()
        }


class LogWindProfile:
    def __init__(self, avg_wind_spd, heights, kappa=0.4, obs_height=None, z0=None):
        self.U = avg_wind_spd
        self.z = heights
        self.log_z = np.log(heights)
        self.kappa = 0.4
        self.obs_height = obs_height
        self.z0 = z0

    def extrap_log_winds_linear(self):
        return Polynomial.fit(self.U[:2], self.log_z[:2], deg=1)

    def linreg_log_height(self):
        return linregress(self.U, self.log_z)

    def z0_est(self):
        p = self.extrap_log_winds_linear()
        return p(0)

    def ustar(self):
        if self.z0 is not None:
            ustar = (self.kappa * self.U[-1]) / (np.log(self.z[-1] / self.z0))
        else:
            ustar = (self.kappa * self.U[-1]) / (np.log(self.z[-1] / np.exp(self.z0_est())))
        return ustar

    def predict_mean_wind_at_height(self):
        return self.ustar() * np.log(self.z[-1] / np.exp(self.z0)) / self.kappa

    def plt_log_wind_profile(self, linear_extrap=True, log_wind=True):
        fig, ax = plt.subplots()
        ax.plot(self.U, self.log_z,
                label="Observed Data", color='k')

        if linear_extrap:
            ax.plot([0, self.U[0]], [self.z0_est(), self.log_z[0]],
                    label="linear fit", linestyle="--",
                    color='dimgray')

            # slope, intercept, _, _, _ = self.linreg_log_height()
            # ax.semilogy([0, self.U[0]], slope*self.log_z + intercept,
            #         label="linear fit", linestyle = "--",
            #         color='dimgray')
            ax.set_xlabel('U winds (m/s)')
            ax.set_ylabel('Log Z height')

        #fig.colorbar(ax)
        return fig, ax

    def shear_stress_ground(self, avg_rho=1.2):
        return self.ustar() ** 2 * avg_rho
