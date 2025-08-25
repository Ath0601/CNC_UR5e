import re
from geometry_msgs.msg import Pose
from math import pi

class Parser:
    def __init__(self, gcode_file):
        self.gcode_file = gcode_file
        self.scale = 1.0  # default: assume mm
        self.blocks = []
        self._load_blocks()

    def _load_blocks(self):
        with open(self.gcode_file, 'r') as f:
            lines = f.readlines()
        # Remove comments and blank lines, parse into blocks
        for line in lines:
            line = line.split(';')[0].strip()
            if not line:
                continue
            m = re.match(r'([GMT])(\d+)(.*)', line, re.I)
            if m:
                code, num, params_str = m.groups()
                params = self._parse_params(params_str)
                self.blocks.append(((code.upper(), int(num)), params))

    def _parse_params(self, s):
        params = {}
        for part in s.split():
            if len(part) < 2:
                continue
            key, val = part[0].upper(), part[1:]
            try:
                params[key] = float(val)
            except Exception:
                continue
        return params

    def parse(self):
        poses = []
        last_x, last_y, last_z = 0.0, 0.0, 0.0
        default_z = 0.2  # Try a safe default in workspace!

        for block, params in self.blocks:
            code, num = block

            # Set units
            if code == 'G' and num == 21:
                self.scale = 0.001  # mm to m
            if code == 'G' and num == 20:
                self.scale = 0.0254  # inch to m

            # Only append pose if X, Y, or Z is present
            if code == 'G' and num == 1:
                has_xyz = any(axis in params for axis in ['X', 'Y', 'Z'])
                if has_xyz:
                    x = params.get('X', last_x)
                    y = params.get('Y', last_y)
                    z = params.get('Z', last_z)
                    pose = Pose()
                    pose.position.x = x * self.scale
                    pose.position.y = y * self.scale
                    # --- FIX Z LOGIC ---
                    if z is not None:
                        pose.position.z = z * self.scale
                    else:
                        pose.position.z = last_z * self.scale if last_z is not None else default_z
                    # Default orientation (tool pointing down)
                    pose.orientation.x = 0.0
                    pose.orientation.y = 0.0
                    pose.orientation.z = 0.0
                    pose.orientation.w = 1.0
                    poses.append(pose)
                    last_x, last_y, last_z = x, y, z
        # Print all poses for debug!
        print("Parsed poses:")
        for i, pose in enumerate(poses):
            print(f"Pose {i}: x={pose.position.x:.3f}, y={pose.position.y:.3f}, z={pose.position.z:.3f}")
        return poses

