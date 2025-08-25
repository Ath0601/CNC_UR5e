import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from tf2_ros import TransformListener, Buffer
from tf_transformations import quaternion_matrix, quaternion_from_matrix
import numpy as np
from std_msgs.msg import Float64
from scipy.spatial.transform import Rotation as R
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# Use your actual parser
from trialpkg.gcodeparser import GCodeParser

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

        # Sub & Pubs
        self.create_subscription(PoseStamped, '/workpiece_pose', self.vision_pose_callback, 10)
        self.workpiece_pose_base_pub = self.create_publisher(PoseStamped, '/workpiece_pose_base', 10)
        self.gcode_poses_base_pub = self.create_publisher(PoseArray, '/gcode_poses_base', pose_array_qos)
        self.feed_pub = self.create_publisher(Float64, '/gcode_feed_rate', pose_array_qos)
        self.republish_timer = self.create_timer(2.0, self.republish_gcode_poses)

        self.current_workpiece_pose_base = None
        self.gcode_parsed = False
        self.last_gcode_posearray = None
        self.republish_count = 0

    def vision_pose_callback(self, msg):
        try:
            # Transform from camera to base_link
            transform = self.tf_buffer.lookup_transform('base_link', msg.header.frame_id, rclpy.time.Time())
            pose_in_base = self.manual_transform_pose(msg, transform)

            # Apply your workpiece y/z mounting offset, if any
            pose_in_base.pose.position.y = float(pose_in_base.pose.position.y) - 0.168
            pose_in_base.pose.position.z = float(pose_in_base.pose.position.z) + 0.004
            self.get_logger().info(f'The workpiece corner in base frame: x: {pose_in_base.pose.position.x}, y: {pose_in_base.pose.position.y}, z: {pose_in_base.pose.position.z}')
            self.get_logger().info(f'The orientation of the block is: x: {pose_in_base.pose.orientation.x}, y: {pose_in_base.pose.orientation.y}, z: {pose_in_base.pose.orientation.z}, w: {pose_in_base.pose.orientation.w}')

            # Surface pose for G-code
            surface_pose = Pose()
            surface_pose.position.x = pose_in_base.pose.position.x
            surface_pose.position.y = pose_in_base.pose.position.y
            surface_pose.position.z = pose_in_base.pose.position.z - 0.004  # Remove hover offset
            surface_pose.orientation = pose_in_base.pose.orientation

            self.current_workpiece_pose_base = pose_in_base.pose
            self.current_workpiece_pose_surface = surface_pose
            self.workpiece_pose_base_pub.publish(pose_in_base)
            self.get_logger().info("Published workpiece_pose_base")

            # Only parse and transform G-code ONCE, after first good pose
            if not self.gcode_parsed:
                self.parse_and_publish_gcode()
                self.gcode_parsed = True

        except Exception as e:
            self.get_logger().error(f"Transform failed: {e}")

    def manual_transform_pose(self, pose_msg, tf):
        translation = np.array([
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z
        ])
        quat = [
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w
        ]
        T = quaternion_matrix(quat)
        pos = np.array([
            pose_msg.pose.position.x,
            pose_msg.pose.position.y,
            pose_msg.pose.position.z,
            1.0
        ])
        trans_pos = np.dot(T, pos)[:3] + translation

        pose_quat = [
            pose_msg.pose.orientation.x,
            pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z,
            pose_msg.pose.orientation.w
        ]
        rot = quaternion_matrix(pose_quat)
        new_rot = np.dot(T, rot)
        new_quat = quaternion_from_matrix(new_rot)

        pose_out = PoseStamped()
        pose_out.header.frame_id = tf.header.frame_id
        pose_out.header.stamp = self.get_clock().now().to_msg()
        pose_out.pose.position.x = float(trans_pos[0])
        pose_out.pose.position.y = float(trans_pos[1])
        pose_out.pose.position.z = float(trans_pos[2])
        pose_out.pose.orientation.x = float(new_quat[0])
        pose_out.pose.orientation.y = float(new_quat[1])
        pose_out.pose.orientation.z = float(new_quat[2])
        pose_out.pose.orientation.w = float(new_quat[3])
        return pose_out

    def parse_and_publish_gcode(self):
        gcode_path = self.get_parameter('gcode_file').get_parameter_value().string_value
        if not gcode_path:
            self.get_logger().error("No gcode_file parameter set.")
            return

        # Parse waypoints using your parser (these are relative to workpiece corner)
        parser = GCodeParser(gcode_path)
        poses, feed_rate = parser.parse()
        
        if self.current_workpiece_pose_base is None:
            self.get_logger().error("No workpiece pose yet, cannot transform G-code.")
            return

        # Use workpiece pose as origin
        origin = np.array([
            self.current_workpiece_pose_surface.position.x,
            self.current_workpiece_pose_surface.position.y - 0.050,
            self.current_workpiece_pose_surface.position.z
        ])
        q = R.from_euler('xyz', [0, 0, 0]).as_quat()
        self.get_logger().info(f"Workpiece surface origin for G-code: x={origin[0]}, y={origin[1]}, z={origin[2]}")
        self.get_logger().info(f"Workpiece orientation quaternion: x={q[0]}, y={q[1]}, z={q[2]}, w={q[3]}")
        rot = R.from_quat(q)

        transformed_poses = []
        # TOOL DOWN orientation: (-pi/2, 0, 0) in base_link
        tool_down_quat = R.from_euler('xyz', [-np.pi/2, 0, 0]).as_quat()

        for p in poses:
            local = np.array([p.position.x, p.position.y, p.position.z])
            world = rot.apply(local) + origin
            p_base = Pose()
            p_base.position.x, p_base.position.y, p_base.position.z = world
            p_base.orientation.x = float(tool_down_quat[0])
            p_base.orientation.y = float(tool_down_quat[1])
            p_base.orientation.z = float(tool_down_quat[2])
            p_base.orientation.w = float(tool_down_quat[3])
            transformed_poses.append(p_base)
            self.get_logger().info(f'Transformed waypoints in base: x: {p_base.position.x}, y: {p_base.position.y}, z: {p_base.position.z}')

        pa = PoseArray()
        pa.header.frame_id = "base_link"
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.poses = transformed_poses

        feed_msg = Float64()
        feed_msg.data = feed_rate
        self.feed_pub.publish(feed_msg)
        self.last_feed_rate = feed_msg
        self.get_logger().info(f"Published feed rate: {feed_rate} mm/min")

        self.gcode_poses_base_pub.publish(pa)
        self.last_gcode_posearray = pa
        self.get_logger().info(f"Published {len(transformed_poses)} G-code poses in base_link frame.")

    def republish_gcode_poses(self):
        #Periodically re-publish last G-code PoseArray for downstream nodes.
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
