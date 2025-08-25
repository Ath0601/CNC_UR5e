import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from moveit_msgs.action import ExecuteTrajectory
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from scipy.spatial.transform import Rotation as R
import numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from scipy.interpolate import CubicSpline
from moveit_msgs.srv import GetPositionIK
import csv
import matplotlib.pyplot as plt

class CartesianTrajectoryNode(Node):
    def __init__(self):
        super().__init__('workpiece_cartesian_traj_node')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )

        self.exec_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self.pose = None
        self.first_pose_received = False
        self.first_joint_received = False
        self.last_joint_state = None
        self.trajectory_started = False
        self.recording_active = False
        self.joint_efforts = []
        self.joint_velocities = []

        self.pose_subscription = self.create_subscription(PoseStamped, '/workpiece_pose_base', self.pose_callback, 10)
        self.js_subscription = self.create_subscription(JointState, '/joint_states', self.joint_state_cb, qos)

        self.gcode_sub = self.create_subscription(PoseArray, '/gcode_poses_base', self.gcode_cb, 10)
        self.gcode_waypoints = []
        self.gcode_received = False

        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.get_logger().info("Waiting for compute_ik service...")
        self.ik_client.wait_for_service()
        self.get_logger().info("IK service is available.")

        self.get_logger().info("Waiting for execute_trajectory action server...")
        self.exec_client.wait_for_server()
        self.get_logger().info("MoveIt execution action is available.")

    def joint_state_cb(self, msg):
        self.last_joint_state = msg
        if self.recording_active:  # only log during execution
            now_sec = self.get_clock().now().nanoseconds * 1e-9
            self.joint_efforts.append([now_sec] + list(msg.effort))
            self.joint_velocities.append([now_sec] + list(msg.velocity))
        if not self.first_joint_received:
            self.first_joint_received = True
            self.get_logger().info("First joint_states received.")
        self.check_and_start()

    def pose_callback(self, msg):
        if not self.first_pose_received:
            self.pose = msg
            self.first_pose_received = True
            self.get_logger().info("First workpiece_pose_base received.")
        self.check_and_start()

    def gcode_cb(self, msg):
        self.gcode_waypoints = msg.poses
        self.gcode_received = True
        self.get_logger().info(f"Received {len(msg.poses)} G-code waypoints.")

    def check_and_start(self):
        if self.first_joint_received and self.first_pose_received and not self.trajectory_started:
            self.trajectory_started = True
            self.plan_hover_then_descend()

    def plan_hover_then_descend(self):
        msg = self.pose
        pose_goal = Pose()
        pose_goal.position = msg.pose.position
        quat = R.from_euler('xyz', [-np.pi/2, 0, 0]).as_quat()
        pose_goal.orientation.x = quat[0]
        pose_goal.orientation.y = quat[1]
        pose_goal.orientation.z = quat[2]
        pose_goal.orientation.w = quat[3]
        self.get_logger().info(f'The Goal Pose points are: x: {pose_goal.position.x}, y: {pose_goal.position.y}, z:{pose_goal.position.z}')

        q_start = np.array(self.last_joint_state.position)
        self.get_logger().info("Planning hover -> descend...")

        descend_pose = Pose()
        descend_pose.position.x = pose_goal.position.x
        descend_pose.position.y = pose_goal.position.y
        descend_pose.position.z = pose_goal.position.z - 0.004
        descend_pose.orientation = pose_goal.orientation
        self.get_logger().info(f'The descended goal Pose points are: x: {descend_pose.position.x}, y: {descend_pose.position.y}, z:{descend_pose.position.z}')

        self.call_ik_and_exec(q_start, descend_pose, callback=self.after_descend)

    def after_descend(self):
        self.get_logger().info("Descended to workpiece surface. Waiting for G-code waypoints...")
        self.gcode_timer = self.create_timer(1.0, self.try_execute_gcode_path)

    def try_execute_gcode_path(self):
        if self.gcode_received and self.gcode_waypoints:
            self.get_logger().info("Executing G-code Cartesian path!")
            self.plan_gcode_cartesian_path()
            if hasattr(self, 'gcode_timer'):
                self.gcode_timer.cancel()

    def plan_gcode_cartesian_path(self):
        q_start = np.array(self.last_joint_state.position)
        poses = self.gcode_waypoints
        self.get_logger().info(f'The GCode Waypoints: {poses}')

        self.traj_msg = JointTrajectory()
        self.traj_msg.joint_names = self.last_joint_state.name

        self.t_points = [0]
        self.q_mat = [q_start]
        self.time_step = 2.0
        self.prev_q = q_start
        self.ik_index = 0

        self.solve_next_ik(poses)

    def solve_next_ik(self, poses):
        if self.ik_index >= len(poses):
            self.finish_and_execute_cartesian()
            return

        p = poses[self.ik_index]
        ik_pose = Pose()
        ik_pose.position = p.position
        ik_pose.orientation = p.orientation
        self.get_logger().info(f"Solving IK for waypoint {self.ik_index}: {ik_pose}")

        self.solve_ik_async(ik_pose, self.prev_q, lambda q_next: self.after_ik_result(q_next, poses))

    def after_ik_result(self, q_next, poses):
        if q_next is not None:
            self.q_mat.append(q_next)
            self.t_points.append(self.t_points[-1] + self.time_step)
            self.prev_q = q_next
        else:
            self.get_logger().warn(f"Skipping unreachable waypoint {self.ik_index}")

        self.ik_index += 1
        self.solve_next_ik(poses)

    def finish_and_execute_cartesian(self):
        if len(self.q_mat) < 2:
            self.get_logger().error("No valid waypoints to execute.")
            return

        q_mat = np.vstack(self.q_mat)
        spline = CubicSpline(self.t_points, q_mat, axis=0)
        times = np.linspace(0, self.t_points[-1], int(self.t_points[-1]*10))
        for t in times:
            point = JointTrajectoryPoint()
            point.positions = spline(t).tolist()
            point.velocities = spline.derivative(1)(t).tolist()
            point.accelerations = spline.derivative(2)(t).tolist()
            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = int((t - int(t)) * 1e9)
            self.traj_msg.points.append(point)

        self.recording_active = True
        goal_msg = ExecuteTrajectory.Goal()
        goal_msg.trajectory.joint_trajectory = self.traj_msg

        self.get_logger().info("Sending G-code joint-space trajectory to execution (async)...")
        send_future = self.exec_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self.exec_result_callback)

    def call_ik_and_exec(self, q_start, pose_goal, callback=None):
        req = GetPositionIK.Request()
        req.ik_request.group_name = "manipulator"
        req.ik_request.pose_stamped.header.frame_id = "base_link"
        req.ik_request.pose_stamped.pose = pose_goal
        req.ik_request.timeout.sec = 5
        req.ik_request.robot_state.joint_state = self.last_joint_state

        future = self.ik_client.call_async(req)
        def _cb(fut):
            result = fut.result()
            if result is None or not result.solution.joint_state.position:
                self.get_logger().error("IK failed: No solution found for descend!")
                return
            q_goal = np.array(result.solution.joint_state.position)
            self.get_logger().info(f"Descend joint target: {q_goal}")

            t_points = [0, 2]
            q_mat = np.vstack([q_start, q_goal])
            spline = CubicSpline(t_points, q_mat, axis=0)
            times = np.linspace(0, 2, 30)
            traj_msg = JointTrajectory()
            traj_msg.joint_names = self.last_joint_state.name
            for t in times:
                point = JointTrajectoryPoint()
                point.positions = spline(t).tolist()
                point.velocities = spline.derivative(1)(t).tolist()
                point.accelerations = spline.derivative(2)(t).tolist()
                point.time_from_start.sec = int(t)
                point.time_from_start.nanosec = int((t - int(t)) * 1e9)
                traj_msg.points.append(point)

            goal_msg = ExecuteTrajectory.Goal()
            goal_msg.trajectory.joint_trajectory = traj_msg

            self.get_logger().info("Sending descend trajectory to execution (async)...")
            send_future = self.exec_client.send_goal_async(goal_msg)
            send_future.add_done_callback(lambda fut: self._descend_done(fut, callback))
        future.add_done_callback(_cb)

    def _descend_done(self, future, callback):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Descend trajectory goal rejected!")
            return
        self.get_logger().info("Descend trajectory accepted, waiting for result...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda fut: self._descend_exec_done(fut, callback))

    def _descend_exec_done(self, future, callback):
        self.get_logger().info("Descend execution finished.")
        if callback:
            callback()

    def solve_ik_async(self, pose_goal, seed_q, done_cb):
        req = GetPositionIK.Request()
        req.ik_request.group_name = "manipulator"
        req.ik_request.pose_stamped.header.frame_id = "base_link"
        req.ik_request.pose_stamped.pose = pose_goal
        req.ik_request.timeout.sec = 2

        js = JointState()
        js.name = self.last_joint_state.name
        js.position = seed_q.tolist()
        req.ik_request.robot_state.joint_state = js

        future = self.ik_client.call_async(req)
        def _cb(fut):
            result = fut.result()
            if result is None or not result.solution.joint_state.position:
                done_cb(None)
            else:
                done_cb(np.array(result.solution.joint_state.position))
        future.add_done_callback(_cb)

    def exec_result_callback(self, future):
        try:
            goal_handle = future.result()
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
        self.recording_active = False
        with open('joint_efforts.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time'] + self.last_joint_state.name)
            writer.writerows(self.joint_efforts)

        with open('joint_velocities.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time'] + self.last_joint_state.name)
            writer.writerows(self.joint_velocities)

        # Plot efforts
        times = [row[0] for row in self.joint_efforts]
        for j in range(len(self.last_joint_state.name)):
            plt.plot(times, [row[j+1] for row in self.joint_efforts], label=f"Joint {j+1}")
        plt.xlabel('Time (s)')
        plt.ylabel('Effort (Nm)')
        plt.title('Joint Efforts During Milling')
        plt.legend()
        plt.savefig('joint_efforts_plot.png')
        plt.close()
        self.destroy_node()

def main():
    rclpy.init()
    node = CartesianTrajectoryNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
