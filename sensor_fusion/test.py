import numpy 

def sph_to_cart(epsilon, alpha, r):

    """
    Transform sensor readings to Cartesian coordinates in the sensor
    frame. The values of epsilon and alpha are given in radians, while 
    r is in metres. Epsilon is the elevation angle and alpha is the
    azimuth angle (i.e., in the x,y plane).
    """
    p = numpy.zeros(3)  # Position vector 
    p[0] = r * numpy.cos(alpha) * numpy.cos(epsilon)
    p[1] = r * numpy.sin(alpha) * numpy.cos(epsilon)
    p[2] = r * numpy.sin(epsilon) 
    
    # Your code here
    
    return p
  
def estimate_params(P):
    """
    Estimate parameters from sensor readings in the Cartesian frame.
    Each row in the P matrix contains a single 3D point measurement;
    the matrix P has size n x 3 (for n points). The format is:
    
    P = [[x1, y1, z1],
        [x2, x2, z2], ...]
        
    where all coordinate values are in metres. Three parameters are
    required to fit the plane, a, b, and c, according to the equation
    
    z = a + bx + cy
    
    The function should retrn the parameters as a NumPy array of size
    three, in the order [a, b, c].
    """
    param_est = numpy.zeros(3)
    A = numpy.ones((len(P),3))
    b = numpy.ones(len(P))
    for i in range(len(P)):
        temp = P[i] 
        A[i,1:3] = temp[0:2]
        b[i] = temp[2]

    param_est = numpy.linalg.inv(A.T.dot(A)).dot(A.T.dot(b)) 
    
    return param_est
  
if __name__ == "__main__":
    P = [[1, 2, 3],
       [4, 5, 6]]
       
    estimate_params(P)

    print(sph_to_cart(numpy.deg2rad(5), numpy.deg2rad(10), 4))