import numpy as np
import sys

# UR5e DH Parameters
d1 = 0.1625
a2 = -0.425
a3 = -0.3922
d4 = 0.1333
d5 = 0.0997
d6 = 0.0996

def ur5e_ik(x, y, z):
    # "Tool down" orientation: R = identity, so wrist position is:
    wx = x
    wy = y
    wz = z - d6  # subtract last link offset in z

    # Joint 1
    q1 = np.arctan2(wy, wx)

    # Wrist center relative to base frame
    r = np.sqrt(wx**2 + wy**2)
    s = wz - d1

    # Position of wrist center projected onto the base plane
    D = (r**2 + s**2 - a2**2 - a3**2) / (2 * a2 * a3)

    if np.abs(D) > 1.0:
        raise ValueError("Unreachable position (D > 1)")

    q3 = np.arccos(D)
    q2 = np.arctan2(s, r) - np.arctan2(a3 * np.sin(q3), a2 + a3 * np.cos(q3))

    # For "tool down", the last three joints:
    q4 = 0
    q5 = 0
    q6 = 0

    return [q1, q2, q3, q4, q5, q6]

if __name__ == "__main__":

    if len(sys.argv) == 4:
        x, y, z = map(float, sys.argv[1:4])
    else:
        print("Usage: python ur5e_ik.py x y z")
        print("Defaulting to (0.4, 0, 0.2)")
        x, y, z = 0.4, 0.0, 0.2
    try:
        joints = ur5e_ik(x, y, z)
        print("Joint values (radians):", joints)
        print("Joint values in degrees:", np.rad2deg(joints))
    except Exception as e:
        print("IK Error:", e)
