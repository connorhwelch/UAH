import bl_functions as blf
from bl_functions import PerturbationVariable
import numpy as np

# calculate the perturbation variables using the PerturbationVariable dataclass
vert_motion = PerturbationVariable()
potential_temp = PerturbationVariable()

# insert instant obs
w_obs = np.array([0.5, -0.5, 1.0, 0.8, 0.9, -0.2, -0.5, 0.0, -0.9, -0.1], dtype=np.float32)
theta_obs = np.array([295, 293, 295, 298, 292, 294, 292, 289, 293, 299], dtype=np.float32)

# make the dataclasses aware of these obs
vert_motion.add_observation(w_obs)
potential_temp.add_observation(theta_obs)

# claculate the perturbation var
w_prime = vert_motion.perturbations
theta_prime = potential_temp.perturbations
#print('w_prime and theta_prime.........')
wp = w_prime.round(3)
tp = theta_prime.round(3)

# calculate the prime squared values
#print('w_prime squared and theta_prime squared.......')
wp2 = np.round(w_prime**2, 3)
tp2 = np.round(theta_prime**2, 3)

# calc w * theta
#print('w obs times theta obs...........')
wt = np.round(w_obs * theta_obs, 3)

# calc w' * theta'
#print('wprime times theta prime ....... ')

wptp = np.round(vert_motion.perturbations * potential_temp.perturbations, 3)

print(*(f"{wp[x]} & {tp[x]} & {wp2[x]} & {tp2[x]} & {wt[x]} & {wptp[x]}" for x in range(0,len(w_prime))), sep="\n")
print((f"avgs {vert_motion.mean} & {potential_temp.mean} & {np.average(wp).round(3)} & {np.average(tp).round(3)} & {np.average(wp2).round(3)} & "
       f"{np.average(tp2).round(3)} & {np.average(wt).round(3)} & {np.average(wptp).round(3)}")
)
wpm = vert_motion.mean
tpm = potential_temp.mean
print(f'{np.mean(wt)} = {np.round((wpm*tpm),3)} + {np.mean(wptp).round(3)}')

# TAKE THIS STUFF AND MAKE INTO FUNCTIONS OR SOMETHING USEFUL FOR LESS CLUTTER

# Prob 3
print(vert_motion.biased_std,'\n', potential_temp.biased_std)

corr_w_theta = np.corrcoef(w_obs, theta_obs)
print(corr_w_theta[0,1])

