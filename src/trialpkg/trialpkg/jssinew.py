import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose
from tf2_ros import TransformListener, Buffer
from tf_transformations import quaternion_matrix, quaternion_from_matrix
from moveit_msgs.action import ExecuteTrajectory
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from scipy.spatial.transform import Rotation as R
import numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from scipy.interpolate import CubicSpline
from moveit_msgs.srv import GetPositionIK

class CartesianTrajectoryNode(Node):
    def __init__(self):
        super().__init__('workpiece_cartesian_traj_node')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.exec_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')

        self.pose = None
        self.first_pose_received = False
        self.first_joint_received = False
        self.last_joint_state = None
        self.trajectory_started = False
        self.pose_subscription = self.create_subscription(PoseStamped, '/workpiece_pose', self.pose_callback, 10)
        self.js_subscription = self.create_subscription(JointState, '/joint_states', self.joint_state_cb, qos)  # Persistent

        # IK client for /compute_ik
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.get_logger().info("Waiting for compute_ik service...")
        self.ik_client.wait_for_service()
        self.get_logger().info("IK service is available.")

        self.get_logger().info("Waiting for execute_trajectory action server...")
        self.exec_client.wait_for_server()
        self.get_logger().info("MoveIt execution action is available.")

    def joint_state_cb(self, msg):
        self.last_joint_state = msg
        if not self.first_joint_received:
            self.first_joint_received = True
            self.get_logger().info("First joint_states received.")
        if self.first_pose_received and not self.trajectory_started:
            self.trajectory_started = True
            self.plan_and_execute()

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

    def pose_callback(self, msg):
        if not self.first_pose_received:
            self.pose = msg
            self.first_pose_received = True
            self.get_logger().info("First workpiece_pose received.")
        if self.first_joint_received and not self.trajectory_started:
            self.trajectory_started = True
            self.plan_and_execute()

    def plan_and_execute(self):
        msg = self.pose
        self.get_logger().info("Planning trajectory...")
        self.get_logger().info(f"Received workpiece pose in {msg.header.frame_id}: x={msg.pose.position.x:.3f}, y={msg.pose.position.y:.3f}, z={msg.pose.position.z:.3f}")

        try:
            transform = self.tf_buffer.lookup_transform('base_link', msg.header.frame_id, rclpy.time.Time())
            pose_in_base = self.manual_transform_pose(msg, transform)
            self.get_logger().info("Transformed pose to base_link.")
            pose_in_base.pose.position.y = float(pose_in_base.pose.position.y) - 0.168
            pose_in_base.pose.position.z = float(pose_in_base.pose.position.z) + 0.004
            self.get_logger().info(f"After manual adjustment: x={pose_in_base.pose.position.x:.3f}, y={pose_in_base.pose.position.y:.3f}, z={pose_in_base.pose.position.z:.3f}")
        except Exception as e:
            self.get_logger().error(f"Transform failed: {e}")
            return

        quat = R.from_euler('xyz', [-np.pi/2, 0, 0]).as_quat()
        pose_goal = Pose()
        pose_goal.position = pose_in_base.pose.position
        pose_goal.orientation.x = quat[0]
        pose_goal.orientation.y = quat[1]
        pose_goal.orientation.z = quat[2]
        pose_goal.orientation.w = quat[3]

        joint_state_msg = self.last_joint_state
        if not joint_state_msg or not joint_state_msg.name:
            self.get_logger().error("No joint states received yet. Cannot proceed.")
            return
        self.get_logger().info(f"JointState: names={joint_state_msg.name}, positions={joint_state_msg.position}")

        q_start = np.array(joint_state_msg.position)
        self.get_logger().info(f"Trajectory start: {q_start}")

        # Prepare IK request
        req = GetPositionIK.Request()
        req.ik_request.group_name = "manipulator"
        req.ik_request.pose_stamped.header.frame_id = "base_link"
        req.ik_request.pose_stamped.pose = pose_goal
        req.ik_request.timeout.sec = 5
        req.ik_request.timeout.nanosec = 0
        if self.last_joint_state:
            req.ik_request.robot_state.joint_state = self.last_joint_state

        future = self.ik_client.call_async(req)
        future.add_done_callback(lambda fut: self.ik_done_callback(fut, q_start, joint_state_msg))

    def ik_done_callback(self, future, q_start, joint_state_msg):
        result = future.result()
        self.get_logger().info(f"IK response: {result}")
        if result is None or not result.solution.joint_state.position:
            self.get_logger().error("IK failed: No solution found!")
            return
        q_goal = np.array(result.solution.joint_state.position)
        self.get_logger().info(f"Trajectory goal: {q_goal}")

        # Interpolate in joint space with cubic spline
        t_points = [0, 2]  # Start at t=0, end at t=2s (change if needed)
        q_mat = np.vstack([q_start, q_goal])
        spline = CubicSpline(t_points, q_mat, axis=0)

        n_steps = 30
        times = np.linspace(0, 2, n_steps)
        traj_msg = JointTrajectory()
        traj_msg.joint_names = joint_state_msg.name
        self.get_logger().info(f"Trajectory joint names: {traj_msg.joint_names}")
        for t in times:
            point = JointTrajectoryPoint()
            point.positions = spline(t).tolist()
            point.velocities = spline.derivative(1)(t).tolist()
            point.accelerations = spline.derivative(2)(t).tolist()
            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = int((t - int(t)) * 1e9)
            traj_msg.points.append(point)

        # Send to MoveIt ExecuteTrajectory
        goal_msg = ExecuteTrajectory.Goal()
        goal_msg.trajectory.joint_trajectory = traj_msg

        self.get_logger().info("Sending joint-space trajectory to execution (async)...")
        send_future = self.exec_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self.exec_result_callback)
        self.get_logger().info("Goal sent, waiting for exec_result_callback...")

    def exec_result_callback(self, future):
        try:
            goal_handle = future.result()
            self.get_logger().info(f"Goal handle: {goal_handle}")
            if not goal_handle.accepted:
                self.get_logger().error("Trajectory execution goal was rejected.")
                return

            self.get_logger().info("Trajectory accepted, waiting for result...")
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self.execution_done_callback)
        except Exception as e:
            self.get_logger().error(f"exec_result_callback exception: {e}")

    def execution_done_callback(self, future):
        self.get_logger().info("Execution finished.")
        self.destroy_node()

def main():
    rclpy.init()
    node = CartesianTrajectoryNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
