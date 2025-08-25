#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.srv import GetPositionIK


class CornerMoveNode(Node):
    def __init__(self):
        super().__init__('corner_move_node')

        # QoS for /joint_states (latched-ish & reliable)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )

        # Action & service clients
        self.exec_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')

        self.get_logger().info("Waiting for compute_ik service...")
        self.ik_client.wait_for_service()
        self.get_logger().info("IK service is available.")

        self.get_logger().info("Waiting for execute_trajectory action server...")
        self.exec_client.wait_for_server()
        self.get_logger().info("MoveIt execution action is available.")

        # State
        self.pose_base: PoseStamped | None = None
        self.last_joint_state: JointState | None = None
        self.first_pose_received = False
        self.first_joint_received = False
        self.trajectory_started = False

        # Subs
        self.pose_sub = self.create_subscription(
            PoseStamped, '/workpiece_pose_base', self.pose_cb, 10)
        self.js_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_cb, qos)

    # ---------------- Callbacks ----------------
    def pose_cb(self, msg: PoseStamped):
        if not self.first_pose_received:
            self.first_pose_received = True
            self.get_logger().info("First /workpiece_pose_base received.")
        self.pose_base = msg
        self.try_start()

    def joint_state_cb(self, msg: JointState):
        if not self.first_joint_received:
            self.first_joint_received = True
            self.get_logger().info("First /joint_states received.")
        self.last_joint_state = msg
        self.try_start()

    # ---------------- Flow control ----------------
    def try_start(self):
        if (self.first_pose_received and self.first_joint_received
                and not self.trajectory_started):
            self.trajectory_started = True
            self.plan_hover_then_descend_async()

    # ---------------- Planning ----------------
    def plan_hover_then_descend_async(self):
        assert self.pose_base is not None and self.last_joint_state is not None

        # Tool-down hover @ corner (4mm above surface already accounted in your upstream)
        hover_pose = Pose()
        hover_pose.position = self.pose_base.pose.position
        quat = R.from_euler('xyz', [-np.pi/2, 0, 0]).as_quat()
        hover_pose.orientation.x, hover_pose.orientation.y, hover_pose.orientation.z, hover_pose.orientation.w = quat

        # Descend 4mm to touch the surface
        descend_pose = Pose()
        descend_pose.position.x = hover_pose.position.x
        descend_pose.position.y = hover_pose.position.y
        descend_pose.position.z = hover_pose.position.z - 0.004
        descend_pose.orientation = hover_pose.orientation

        q_start = np.array(self.last_joint_state.position)

        # 1) IK to descend pose (async)
        self.solve_ik_async(descend_pose, seed_q=q_start, done_cb=lambda q_goal: self._after_descend_ik(q_start, q_goal))

    def _after_descend_ik(self, q_start: np.ndarray, q_goal: np.ndarray | None):
        if q_goal is None:
            self.get_logger().error("IK failed for descend pose. Aborting.")
            return

        # 2) Build cubic-spline joint trajectory from q_start -> q_goal
        t_knots = [0.0, 2.0]  # 2 seconds
        q_mat = np.vstack([q_start, q_goal])
        spline = CubicSpline(t_knots, q_mat, axis=0)

        times = np.linspace(0.0, t_knots[-1], int(t_knots[-1] * 30))  # 30 Hz discretization
        traj = JointTrajectory()
        traj.joint_names = self.last_joint_state.name
        for t in times:
            pt = JointTrajectoryPoint()
            pt.positions = spline(t).tolist()
            pt.velocities = spline.derivative(1)(t).tolist()
            pt.accelerations = spline.derivative(2)(t).tolist()
            pt.time_from_start.sec = int(t)
            pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
            traj.points.append(pt)

        goal = ExecuteTrajectory.Goal()
        goal.trajectory.joint_trajectory = traj

        # 3) Execute asynchronously
        self.get_logger().info("Sending corner descend trajectory...")
        send_future = self.exec_client.send_goal_async(goal)
        send_future.add_done_callback(self._after_exec_goal_sent)

    def _after_exec_goal_sent(self, fut):
        goal_handle = fut.result()
        if not goal_handle.accepted:
            self.get_logger().error("Corner descend trajectory goal rejected.")
            return
        self.get_logger().info("Corner descend goal accepted; waiting for result...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._after_exec_finished)

    def _after_exec_finished(self, fut):
        self.get_logger().info("Corner move finished.")
        self.destroy_node()

    # ---------------- Helpers ----------------
    def solve_ik_async(self, pose_goal: Pose, seed_q: np.ndarray, done_cb):
        req = GetPositionIK.Request()
        req.ik_request.group_name = "manipulator"
        req.ik_request.pose_stamped.header.frame_id = "base_link"
        req.ik_request.pose_stamped.pose = pose_goal
        req.ik_request.timeout.sec = 5
        req.ik_request.timeout.nanosec = 0

        seed = JointState()
        seed.name = self.last_joint_state.name
        seed.position = seed_q.tolist()
        req.ik_request.robot_state.joint_state = seed

        future = self.ik_client.call_async(req)

        def _cb(f):
            res = f.result()
            if res is None or not res.solution.joint_state.position:
                done_cb(None)
            else:
                done_cb(np.array(res.solution.joint_state.position))

        future.add_done_callback(_cb)


def main():
    rclpy.init()
    node = CornerMoveNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
