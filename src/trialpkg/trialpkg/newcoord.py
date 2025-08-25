#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from tf2_ros import TransformListener, Buffer
from tf_transformations import quaternion_matrix, quaternion_from_matrix, quaternion_multiply
import numpy as np
from std_msgs.msg import Float64
from scipy.spatial.transform import Rotation as R
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# Use your parser (which already maps G-code axes for spindle-on-Y)
from trialpkg.gcodeparser import GCodeParser


def _pose_dist(a: Pose, b: Pose) -> float:
    return float(np.linalg.norm([
        a.position.x - b.position.x,
        a.position.y - b.position.y,
        a.position.z - b.position.z
    ]))

def _dedup_poses(poses, tol=1e-6):
    if not poses:
        return []
    out = [poses[0]]
    for p in poses[1:]:
        if _pose_dist(p, out[-1]) > tol:
            out.append(p)
    return out

def _engagement_lock_local(waypoints_local, dx=0.002):
    """
    Operate in the WORKPIECE LOCAL frame:
      - Find plunge (largest |ΔY_local|)
      - Find radial offset (largest |ΔZ_local|) after plunge
      - Treat last point as end of long X pass
      - Build dense X_local line with Y_local, Z_local locked to values at offset
    """
    n = len(waypoints_local)
    if n < 3:
        return waypoints_local[:]

    dY = [abs(waypoints_local[i].position.y - waypoints_local[i-1].position.y) for i in range(1, n)]
    i_plunge = int(np.argmax(dY)) + 1
    if i_plunge >= n - 1:
        return waypoints_local[:]

    dZ_post = [abs(waypoints_local[i].position.z - waypoints_local[i-1].position.z) for i in range(i_plunge + 1, n)]
    if not dZ_post:
        return waypoints_local[:]
    i_offset = (i_plunge + 1) + int(np.argmax(dZ_post))
    if i_offset >= n - 1:
        return waypoints_local[:]

    i_end = n - 1

    Y_lock = waypoints_local[i_offset].position.y  # axial depth in LOCAL frame
    Z_lock = waypoints_local[i_offset].position.z  # radial offset in LOCAL frame
    x0 = waypoints_local[i_offset].position.x
    x1 = waypoints_local[i_end].position.x
    step = dx if x1 >= x0 else -dx

    dense = []
    # keep originals up to i_offset (exclusive)
    dense.extend(waypoints_local[:i_offset])

    # anchor (start of pass), snap to locked Y/Z
    anchor = Pose()
    anchor.position.x = x0
    anchor.position.y = Y_lock
    anchor.position.z = Z_lock
    anchor.orientation = waypoints_local[i_offset].orientation
    dense.append(anchor)

    # march along X_local with Y/Z held
    x = x0
    while (step > 0 and x + step < x1) or (step < 0 and x + step > x1):
        x += step
        mid = Pose()
        mid.position.x = x
        mid.position.y = Y_lock
        mid.position.z = Z_lock
        mid.orientation = waypoints_local[i_offset].orientation
        dense.append(mid)

    # final point, locked
    endp = Pose()
    endp.position.x = x1
    endp.position.y = Y_lock
    endp.position.z = Z_lock
    endp.orientation = waypoints_local[i_end].orientation
    dense.append(endp)

    return _dedup_poses(dense, tol=1e-8)


class CoordinateTransformNode(Node):
    def __init__(self):
        super().__init__('coordinate_transform_node')

        pose_array_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=5
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.declare_parameter('gcode_file', '/home/atharva/trial_ws/src/trialpkg/gcodefiles/newgcode.gcode')

        # Subs & pubs
        self.create_subscription(PoseStamped, '/workpiece_pose', self.vision_pose_callback, 10)
        self.workpiece_pose_base_pub = self.create_publisher(PoseStamped, '/workpiece_pose_base', 10)
        self.gcode_poses_base_pub = self.create_publisher(PoseArray, '/gcode_poses_base', pose_array_qos)
        self.feed_pub = self.create_publisher(Float64, '/gcode_feed_rate', pose_array_qos)
        self.republish_timer = self.create_timer(2.0, self.republish_gcode_poses)

        self.current_workpiece_pose_base = None
        self.current_workpiece_pose_surface = None
        self.gcode_parsed = False
        self.last_gcode_posearray = None
        self.last_feed_rate = None
        self.republish_count = 0

    def vision_pose_callback(self, msg: PoseStamped):
        try:
            # Transform from camera to base_link
            tf = self.tf_buffer.lookup_transform('base_link', msg.header.frame_id, rclpy.time.Time())
            pose_in_base = self.manual_transform_pose(msg, tf)

            # Mount offsets (as in your file)
            pose_in_base.pose.position.y = float(pose_in_base.pose.position.y) - 0.168
            pose_in_base.pose.position.z = float(pose_in_base.pose.position.z) + 0.004
            self.get_logger().info(f'The workpiece corner in base frame: x: {pose_in_base.pose.position.x}, y: {pose_in_base.pose.position.y}, z: {pose_in_base.pose.position.z}')
            self.get_logger().info(f'The orientation of the block is: x: {pose_in_base.pose.orientation.x}, y: {pose_in_base.pose.orientation.y}, z: {pose_in_base.pose.orientation.z}, w: {pose_in_base.pose.orientation.w}')

            # Surface pose used as local origin
            surface_pose = Pose()
            surface_pose.position.x = pose_in_base.pose.position.x
            surface_pose.position.y = pose_in_base.pose.position.y
            surface_pose.position.z = pose_in_base.pose.position.z - 0.004  # remove hover
            surface_pose.orientation = pose_in_base.pose.orientation

            self.current_workpiece_pose_base = pose_in_base.pose
            self.current_workpiece_pose_surface = surface_pose
            self.workpiece_pose_base_pub.publish(pose_in_base)
            self.get_logger().info("Published workpiece_pose_base")

            if not self.gcode_parsed:
                self.parse_and_publish_gcode()
                self.gcode_parsed = True

        except Exception as e:
            self.get_logger().error(f"Transform failed: {e}")

    def manual_transform_pose(self, pose_msg: PoseStamped, tf):
        translation = np.array([
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z
        ])
        quat_tf = [
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w
        ]
        T_tf = quaternion_matrix(quat_tf)

        pos_cam = np.array([
            pose_msg.pose.position.x,
            pose_msg.pose.position.y,
            pose_msg.pose.position.z,
            1.0
        ])
        pos_base = np.dot(T_tf, pos_cam)[:3] + translation

        quat_cam = [
            pose_msg.pose.orientation.x,
            pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z,
            pose_msg.pose.orientation.w
        ]
        R_cam = quaternion_matrix(quat_cam)
        R_base = np.dot(T_tf, R_cam)
        quat_base = quaternion_from_matrix(R_base)

        out = PoseStamped()
        out.header.frame_id = tf.header.frame_id
        out.header.stamp = self.get_clock().now().to_msg()
        out.pose.position.x = float(pos_base[0])
        out.pose.position.y = float(pos_base[1])
        out.pose.position.z = float(pos_base[2])
        out.pose.orientation.x = float(quat_base[0])
        out.pose.orientation.y = float(quat_base[1])
        out.pose.orientation.z = float(quat_base[2])
        out.pose.orientation.w = float(quat_base[3])
        return out

    def parse_and_publish_gcode(self):
        gcode_path = self.get_parameter('gcode_file').get_parameter_value().string_value
        if not gcode_path:
            self.get_logger().error("No gcode_file parameter set.")
            return

        parser = GCodeParser(gcode_path)
        waypoints_local, feed_rate = parser.parse()   # LOCAL (workpiece) coordinates

        if self.current_workpiece_pose_surface is None:
            self.get_logger().error("No workpiece pose yet, cannot transform G-code.")
            return

        # --- Build base transform from the actual workpiece quaternion (NOT identity) ---
        wp_q = self.current_workpiece_pose_surface.orientation
        q_workpiece = [wp_q.x, wp_q.y, wp_q.z, wp_q.w]
        R44 = quaternion_matrix(q_workpiece)
        R_bw = R44[:3, :3]  # 3x3 base<-workpiece rotation

        origin_base = np.array([
            self.current_workpiece_pose_surface.position.x,
            self.current_workpiece_pose_surface.position.y - 0.050,  # your offset
            self.current_workpiece_pose_surface.position.z
        ], dtype=float)

        self.get_logger().info(f"Workpiece surface origin for G-code (base): x={origin_base[0]:.6f}, y={origin_base[1]:.6f}, z={origin_base[2]:.6f}")
        self.get_logger().info(f"Using workpiece quaternion for transform (base <- block).")

        # --- Keep engagement in LOCAL frame, then transform to base ---
        engaged_local = _engagement_lock_local(waypoints_local, dx=0.002)

        # Tool attitude: your previous fixed tool-down in LOCAL, then rotate into base
        q_tool_local = R.from_euler('xyz', [-np.pi/2, 0, 0]).as_quat()
        q_tool_base  = quaternion_multiply(q_workpiece, q_tool_local)

        transformed_poses = []
        for p in engaged_local:
            p_loc = np.array([p.position.x, p.position.y, p.position.z], dtype=float)
            p_base = origin_base + R_bw.dot(p_loc)

            out = Pose()
            out.position.x = float(p_base[0])
            out.position.y = float(p_base[1])
            out.position.z = float(p_base[2])
            out.orientation.x = float(q_tool_base[0])
            out.orientation.y = float(q_tool_base[1])
            out.orientation.z = float(q_tool_base[2])
            out.orientation.w = float(q_tool_base[3])
            transformed_poses.append(out)
            # Debug the first few points
            if len(transformed_poses) <= 5:
                self.get_logger().info(f'Transformed (base): {out.position.x:.3f}, {out.position.y:.3f}, {out.position.z:.3f}')

        pa = PoseArray()
        pa.header.frame_id = "base_link"
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.poses = transformed_poses

        # Publish feed + poses
        feed_msg = Float64()
        feed_msg.data = float(feed_rate) if feed_rate is not None else 300.0
        self.feed_pub.publish(feed_msg)
        self.last_feed_rate = feed_msg
        self.get_logger().info(f"Published feed rate: {feed_msg.data:.3f} mm/min")

        self.gcode_poses_base_pub.publish(pa)
        self.last_gcode_posearray = pa
        self.get_logger().info(f"Published {len(transformed_poses)} G-code poses (base frame).")

    def republish_gcode_poses(self):
        # Periodically re-publish last G-code PoseArray for downstream nodes.
        if self.last_gcode_posearray and self.last_feed_rate and self.republish_count < 10:
            self.last_gcode_posearray.header.stamp = self.get_clock().now().to_msg()
            self.gcode_poses_base_pub.publish(self.last_gcode_posearray)
            self.feed_pub.publish(self.last_feed_rate)
            self.get_logger().info(f"Re-published /gcode_poses_base (repeat {self.republish_count+1}/10)")
            self.get_logger().info(f"Re-published /gcode_feed_rate (repeat {self.republish_count+1}/10)")
            self.republish_count += 1


def main():
    rclpy.init()
    node = CoordinateTransformNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
