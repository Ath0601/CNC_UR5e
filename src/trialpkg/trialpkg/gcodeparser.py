from pygcode import Line
import numpy as np
from geometry_msgs.msg import Pose
import math
from scipy.spatial.transform import Rotation as R
from trialpkg.urik import ur5e_ik

class GCodeParser:
    def __init__(self, filepath, scale=1.0, debug=False, origin=(0,0,0)):
        self.filepath = filepath
        self.scale = scale
        self.debug = debug
        self.origin = origin
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.feed_rate_mm_min = None

    def euler_to_quat(self, roll, pitch, yaw):
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return qx, qy, qz, qw

    def sample_arc(self, start, end, center, direction, num=20):
        radius = np.linalg.norm(start - center)
        v0 = start - center
        v1 = end - center
        a0 = math.atan2(v0[1], v0[0])
        a1 = math.atan2(v1[1], v1[0])
        if direction == 'CW':
            if a1 > a0:
                a1 -= 2 * math.pi
            angles = np.linspace(a0, a1, num)
        else:
            if a1 < a0:
                a1 += 2 * math.pi
            angles = np.linspace(a0, a1, num)
        points = np.stack((center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)), axis=-1)
        return points

    def arc_center_from_r(self, start, end, r, direction):
        mid = (start[:2] + end[:2]) / 2.0
        chord = end[:2] - start[:2]
        chord_len = np.linalg.norm(chord)
        if chord_len == 0:
            raise ValueError("Start and end for arc are identical!")
        r = abs(r)
        if r < chord_len / 2:
            raise ValueError("Radius too small for arc!")
        h = math.sqrt(r**2 - (chord_len/2)**2)
        perp = np.array([-chord[1], chord[0]])
        perp /= np.linalg.norm(perp)
        if direction == 'CW':
            perp = -perp
        center = mid + perp * h
        return center

    def parse(self):
        waypoints = []
        quat = R.from_euler('xyz', [-np.pi/2, 0, 0]).as_quat()
        orientation = [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
        absolute = True

        with open(self.filepath, 'r') as f:
            for line in f:
                print("[DEBUG] Reading line:", line.strip())
                if line.startswith('G21'):
                    self.scale = 0.001
                if line.startswith('G0') or line.startswith('G1'):
                    words = line.split()
                    x = y = z = None
                    for word in words:
                        if word.startswith('X'):
                            x = float(word[1:]) * self.scale
                        if word.startswith('Y'):
                            y = float(word[1:]) * self.scale
                        if word.startswith('Z'):
                            z = float(word[1:]) * self.scale
                        if word.startswith('F'):
                            self.feed_rate_mm_min = float(word[1:])
                    # Update stored positions (but swap Y/Z meaning for simulation)
                    if x is not None:
                        self.current_x = x
                    if y is not None:
                        self.current_z = y  # G-code Y -> sim Z
                    if z is not None:
                        self.current_y = z  # G-code Z -> sim Y
                    pose = Pose()
                    pose.position.x = self.current_x + self.origin[0]
                    pose.position.y = self.current_y + self.origin[1]
                    pose.position.z = self.current_z + self.origin[2]
                    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = orientation
                    waypoints.append(pose)
        return waypoints, self.feed_rate_mm_min

def main():
    parser = GCodeParser('/home/atharva/trial_ws/src/trialpkg/gcodefiles/newgcode.gcode')
    poses, feed_rate = parser.parse()
    for p in poses:
        x = p.position.x
        y = p.position.y
        z = p.position.z
        ik = np.array(ur5e_ik(x,y,z))
        ikdeg = np.round(np.rad2deg(ik),3)
    print("Parsed waypoints are: ", poses)
    print("Feed rate (mm/min):", feed_rate)
    print("IK Solution for each pose is ", ikdeg)

if __name__ == '__main__':
    main()
