import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

I = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
V = np.array([1.23, 1.38, 2.06, 2.47, 3.17])

plt.scatter(I, V)

plt.xlabel('current (A)')
plt.ylabel('voltage (V)')
plt.grid(True)
plt.show()



## Recursive Solution

# Initialize the 2x1 parameter vector x (i.e., x_0).
#x_k = np.array([ [np.random.normal(4,3)]  , [np.random.normal(0, np.sqrt(0.2))] ]) 
x_k = np.array([4., 0.])


#Initialize the 2x2 covaraince matrix (i.e. P_0). Off-diangonal elements should be zero.
P_k = np.array([[20.0, 0],
               [0, 0.2]])

# Our voltage measurement variance (denoted by R, don't confuse with resistance).
R_k = np.array([[0.0225]])

# Pre allocate space to save our estimates at every step.
num_meas = I.shape[0]
x_hist = np.zeros((num_meas + 1, 2))
P_hist = np.zeros((num_meas + 1, 2, 2))

x_hist[0] = x_k
P_hist[0] = P_k

# Iterate over all the available measurements.
for k in range(num_meas):
    # Construct H_k (Jacobian).
    H_k = np.array([[I[k], 1.]])

    # Construct K_k (gain matrix).
    K_k = P_k @ H_k.T @ inv(H_k @ P_k @ H_k.T + R_k)
    
                    
    # Update our estimate.
    x_k = x_k + K_k @ (V[k] - H_k @ x_k)
    

    # Update our uncertainty (covariance)
    P_k = (np.eye(2) - (K_k @ H_k) ) @P_k   

    # Keep track of our history.
    P_hist[k + 1] = P_k
    x_hist[k + 1] = x_k
    
print('The slope and offset parameters of the best-fit line (i.e., the resistance and offset) are [R, b]:')
print(x_k)
