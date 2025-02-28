import bl_functions
from bl_functions import LogWindProfile
import numpy as np
wind_obs_p2a = [1, 2, 3, 3, 3, 3]
heights_p2a = [1, 10, 100, 200, 500, 1000]
wind_obs_p2b = [1, 4]
heights_p2b = [1, 10]

heights_p9 = [1, 4, 10, 20, 50, 100, 300, 500, 1000, 2000]
wind_obs_p9 = [3.7, 5.0, 5.8, 6.5, 7.4, 8.0, 9.0, 9.5, 10.0, 10.0]

if __name__ == "__main__":
    lp = LogWindProfile(avg_wind_spd=wind_obs_p2a[:2], heights=heights_p2a[:2])
    print(lp.z)
    print(lp.U)
    print(lp.z0_est(), '  ---->  ', np.exp(lp.z0_est()))
    print(lp.linreg_log_height())
    print('ustar  ',lp.ustar())
    fig, ax = lp.plt_log_wind_profile()
    # fig.show()
    # fig.savefig('bl_hw5_p2.png')

    lp = LogWindProfile(avg_wind_spd=wind_obs_p2b, heights=heights_p2b, z0=0.1)
    print(lp.z0_est(), '  ---->  ', np.exp(lp.z0_est()))
    print(lp.linreg_log_height())
    print('ustar  ',lp.ustar())
    fig, ax = lp.plt_log_wind_profile()
    # fig.show()
    # fig.savefig('bl_hw5_p2b.png')

    lp = LogWindProfile(avg_wind_spd=wind_obs_p9, heights=heights_p9, kappa=.35)
    print(lp.z0_est(), '  ---->', np.exp(lp.z0_est()))
    print(lp.linreg_log_height())
    print('ustar  ',lp.ustar())
    print(lp.shear_stress_ground())
    fig, ax = lp.plt_log_wind_profile()
    ax.hlines(y=0.69, xmin=0, xmax=4.3, color="red")
    ax.hlines(y=-2.3, xmin=0, xmax=1.5, color='blue')
    # fig.show()
    # fig.savefig('bl_hw5_p9.png')


