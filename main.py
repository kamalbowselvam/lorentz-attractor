import numpy as np 
import matplotlib.pyplot as plt 


def lorenz(t, y, sigma=10, beta=(8 / 3), rho=28):

    return np.array([sigma * (y[1] - y[0]),
                    y[0] * (rho - y[2]) - y[1],
                    (y[0] * y[1]) - (beta * y[2]),
                    ])

def runge_kutta4(func, tk, yk , dt = 0.01, **kwargs):

    f1 = func(tk, yk, **kwargs)
    f2 = func(tk + dt / 2, yk + (f1 * (dt / 2)), **kwargs)
    f3 = func(tk + dt / 2, yk + (f2 * (dt / 2)), **kwargs)
    f4 = func(tk + dt, yk + (f3 * dt), **kwargs)

    return yk + (dt / 6) * (f1 + (2 * f2) + (2 * f3) + f4)


dt = 0.01 # time stepping 
time = np.arange(0.0, 60.0, dt) # Time vector 
y0 = np.array([-7, 8, 26]) # intial conditoin 
y1 = np.array([7, 8,  27])
state_history1 = [] # Empty list to store state information
state_history2 = [] # Empty list to store state information
yk = y0
yc = y1
t = 0

for t in time:
    # save current state
    state_history1.append(yk)
    state_history2.append(yc)
    yk = runge_kutta4(lorenz, t, yk, dt)
    yc = runge_kutta4(lorenz, t,yc,dt)
state_history1 = np.array(state_history1)
state_history2 = np.array(state_history2)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(state_history1[:,0],'r', lw=.5)
ax.plot(state_history2[:,0],'b', lw=.5)

ax.set_title("Lorenz System Trajectory")
ax.set_xlabel("Time")
ax.set_ylabel('Y')
plt.show()




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(state_history1[:,0],state_history1[:,1],state_history1[:,2],'r', lw=.5)
ax.plot(state_history2[:,0],state_history2[:,1],state_history2[:,2],'b', lw=.5)
ax.set_title("Lorenz System Trajectory")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()