import numpy as np
import sys

# UR5e DH Parameters
DH = [
    [0,        0,      0.1625,  np.pi/2],
    [0,   -0.425,      0,       0],
    [0,   -0.3922,     0,       0],
    [0,        0,      0.1333,  np.pi/2],
    [0,        0,      0.0997, -np.pi/2],
    [0,        0,      0.0996,  0],
]

def dh_transform(theta, a, d, alpha):
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,      sa,     ca,    d],
        [0,       0,      0,    1]
    ])

def ur5e_fk(joints):
    T = np.eye(4)
    for i in range(6):
        T = T @ dh_transform(joints[i], DH[i][1], DH[i][2], DH[i][3])
    return T

if __name__ == "__main__":
    # Example: six zeros
    if len(sys.argv) == 7:
        joints = [float(x) for x in sys.argv[1:7]]
    else:
        print("Usage: python ur5e_fk.py q1 q2 q3 q4 q5 q6 (in radians)")
        print("Defaulting to zeros")
        #joints = [0, -np.pi/2, (2*np.pi)/3, 0, np.pi/2, 0]
        joints = [np.deg2rad(-10),np.deg2rad(-56),np.deg2rad(98),np.deg2rad(-122),np.deg2rad(-90),0]
        print("Joint values are: ",joints)
    T = ur5e_fk(joints)
    print("End Effector Pose (Homogeneous Transform):\n", T)
    print("Position (x, y, z):", T[:3, 3])
