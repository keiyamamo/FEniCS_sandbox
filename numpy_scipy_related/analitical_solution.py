import numpy as np
from scipy.integrate import odeint 
import matplotlib.pyplot as plt
import pandas as pd

# Initialization 
tstart = 0 
tstop = 0.5 
increment = 0.001

# Initial condition 
x_init = [0,0]

t = np.arange(tstart,tstop+increment,increment)

# Function that returns dx/dt 
def mydiff(x, t, params_dict):
    F = params_dict['F']
    A = params_dict['A']
    c = params_dict['c']
    k = params_dict['k']
    m = params_dict['m']

    dx1dt = x[1]
    dx2dt = (F - c*x[1]*A - k*x[0]*A)/m

    dxdt = [dx1dt, dx2dt] 
    return dxdt 

# Solve ODE
params_dict = {'c': 1E2, 'k': 1E5, 'rho': 1.0E3, 'V': 4.1167E-7, 'm': 4.1167E-7*1.0E3, 'gravity': -2, 'F': -2*4.1167E-7*1.0E3, 'A': 2.2E-4}
x = odeint(mydiff, x_init, t, args=(params_dict,))

x1 = x[:,0]
x2 = x[:,1]
from IPython import embed; embed(); exit(1)
# # Plot the Results
# plt.plot(t,x1)
# #plt.plot(t,x2)
# plt.title('Simulation of Mass-Spring-Damper System') 
# plt.xlabel('t')
# plt.ylabel('x(t)') 
# plt.legend(["x1", "x2"]) 
# plt.grid()
# plt.show()


# # dictionary of lists  
# dict = {'disp': x1, 'vel': x2}  
# df = pd.DataFrame(dict)  
# # saving the dataframe 
# df.to_csv('analitical.csv') 