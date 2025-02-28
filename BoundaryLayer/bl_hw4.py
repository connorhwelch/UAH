import bl_functions as blf
import bl_data as bld
from bl_functions import RichardsonNumber

if __name__ == "__main__":

    data = bld.data_hw4q14
    print(type(data['Theta_bar']))
    data['Theta_bar'] = data['Theta_bar'][::-1]
    data["height"] = data["height"][::-1]
    data["U_bar"] = data["U_bar"][::-1]

    print(data["height"])
    for i in range(0, len(data['Theta_bar'])-1):
        f = i+1
        ri_calc = RichardsonNumber(theta_i=data['Theta_bar'][i],
                                   theta_f=data['Theta_bar'][f],
                                   height_i=data['height'][i],
                                   height_f=data['height'][f],
                                   uwind_i=data['U_bar'][i],
                                   uwind_f=data['U_bar'][f],
                                   vwind_i=0,
                                   vwind_f=0
                                   )
        print(ri_calc.bulk_richardson())

    data = {"height": [0, 100, 300, 500, 900, 1000, 1500],
            "Theta_bar": [291, 290, 290, 292, 292, 293, 294],
            "U_bar": [0, 3, 3, 4, 5, 9, 10]}
    print("--------------------------------------")
    for i in range(0, len(data['Theta_bar']) - 1):
        f = i + 1
        ri_calc = RichardsonNumber(theta_i=data['Theta_bar'][i],
                                   theta_f=data['Theta_bar'][f],
                                   height_i=data['height'][i],
                                   height_f=data['height'][f],
                                   uwind_i=data['U_bar'][i],
                                   uwind_f=data['U_bar'][f],
                                   vwind_i=0,
                                   vwind_f=0
                                   )
        print(ri_calc.bulk_richardson())