#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK

from scipy.spatial.transform import Rotation as R
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class IKAndTrajectoryNode(Node):
    def __init__(self):
        super().__init__('ik_and_trajectory_node')

        # Tool-down orientation
        self.tool_down_quat = R.from_euler('xyz', [-np.pi/2, 0, 0]).as_quat()

        # Controller/MoveIt group joint order
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        # State
        self.pose_base: PoseStamped | None = None
        self.last_joint_state: JointState | None = None
        self.first_pose_received = False
        self.first_joint_received = False
        self.trajectory_started = False

        # QoS for /joint_states (keep last sample for late joiners)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )

        # Subscriptions
        self.create_subscription(PoseStamped, '/workpiece_pose_base', self.pose_cb, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, qos)

        # IK service client
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')

        # Trajectory publisher (keep your original topic)
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        self.get_logger().info("IK and trajectory node ready.")

    # ---------- Helpers ----------
    def _reorder(self, vec, names_from, names_to):
        idx = [names_from.index(n) for n in names_to]
        return np.asarray([vec[i] for i in idx], dtype=float)

    @staticmethod
    def _wrap_to_pi(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _nearest_angles(self, q_from, q_to):
        q_from = np.asarray(q_from, dtype=float)
        q_to   = np.asarray(q_to, dtype=float)
        q_adj = np.empty_like(q_to)
        for i in range(q_to.size):
            delta = self._wrap_to_pi(q_to[i] - q_from[i])
            q_adj[i] = q_from[i] + delta
        return q_adj

    @staticmethod
    def _make_seed_js(names, positions_np):
        js = JointState()
        js.name = list(names)
        js.position = np.asarray(positions_np, dtype=float).tolist()
        return js

    # ---------- Callbacks ----------
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

    # ---------- Flow control ----------
    def try_start(self):
        if (self.first_pose_received and self.first_joint_received
                and not self.trajectory_started):
            self.trajectory_started = True
            self.plan_hover_then_target()

    # ---------- Planning ----------
    def plan_hover_then_target(self):
        assert self.pose_base is not None and self.last_joint_state is not None

        # Approach (hover) 4 mm above Z
        approach_pose = PoseStamped()
        approach_pose.header.frame_id = 'base_link'
        approach_pose.pose.position.x = self.pose_base.pose.position.x
        approach_pose.pose.position.y = self.pose_base.pose.position.y
        approach_pose.pose.position.z = self.pose_base.pose.position.z + 0.004
        approach_pose.pose.orientation.x = self.tool_down_quat[0]
        approach_pose.pose.orientation.y = self.tool_down_quat[1]
        approach_pose.pose.orientation.z = self.tool_down_quat[2]
        approach_pose.pose.orientation.w = self.tool_down_quat[3]

        # Target pose: at corner (same orientation)
        target_pose = PoseStamped()
        target_pose.header.frame_id = 'base_link'
        target_pose.pose = approach_pose.pose
        target_pose.pose.position.z = self.pose_base.pose.position.z

        # Current joints reordered to group order
        q_start = self._reorder(self.last_joint_state.position,
                                self.last_joint_state.name,
                                self.joint_names)

        # Approach IK seeded with current state
        seed_app = self._make_seed_js(self.joint_names, q_start)
        self.solve_ik_async(approach_pose, seed_app,
                            lambda q_app_raw: self._after_approach_ik(q_start, q_app_raw, target_pose))

    def _after_approach_ik(self, q_start, q_app_raw, target_pose):
        if q_app_raw is None:
            self.get_logger().error("IK failed for approach pose.")
            return

        # Choose nearest-angle version of approach solution
        q_app = self._nearest_angles(q_start, q_app_raw)

        # Target IK seeded with approach solution
        seed_tgt = self._make_seed_js(self.joint_names, q_app)
        self.solve_ik_async(target_pose, seed_tgt,
                            lambda q_tgt_raw: self.generate_and_publish_spline(q_start, q_app, q_tgt_raw))

    # ---------- IK helpers ----------
    def solve_ik_async(self, pose: PoseStamped, seed_state: JointState, done_cb):
        if not self.ik_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('/compute_ik service not available')
            done_cb(None)
            return

        req = GetPositionIK.Request()
        req.ik_request.group_name = 'manipulator'
        req.ik_request.pose_stamped = pose
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout.sec = 1
        req.ik_request.robot_state.joint_state = seed_state

        future = self.ik_client.call_async(req)
        future.add_done_callback(lambda fut: self.handle_ik_response(fut, done_cb))

    def handle_ik_response(self, future, callback):
        try:
            response = future.result()
            if response and response.error_code.val == response.error_code.SUCCESS:
                names_from = list(response.solution.joint_state.name)
                pos_from   = list(response.solution.joint_state.position)
                # Reorder to our group order
                joints = self._reorder(pos_from, names_from, self.joint_names)
                self.get_logger().info(f"IK success (reordered): {joints.tolist()}")
                callback(joints)
            else:
                code = None if not response else response.error_code.val
                self.get_logger().error(f"IK failed with code {code}")
                callback(None)
        except Exception as e:
            self.get_logger().error(f"IK service call failed: {e}")
            callback(None)

    # ---------- PCHIP cubic, 2 segments ----------
    def generate_and_publish_spline(self, q_start, q_app, q_tgt_raw):
        if q_app is None or q_tgt_raw is None:
            self.get_logger().error("Cannot send trajectory: IK failed.")
            return

        # Make target the nearest continuation from approach
        q_tgt = self._nearest_angles(q_app, q_tgt_raw)

        hz = 30.0
        seg = 2.0  # seconds per segment

        # Segment 1: q_start -> q_app over [0, 2]
        t1 = np.array([0.0, seg])
        q1 = np.vstack([q_start, q_app])
        pchip1 = PchipInterpolator(t1, q1, axis=0)

        # Segment 2: q_app -> q_tgt over [2, 4]
        t2_local = np.array([0.0, seg])
        q2 = np.vstack([q_app, q_tgt])
        pchip2 = PchipInterpolator(t2_local, q2, axis=0)

        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        # Sample segment 1
        times1 = np.linspace(0.0, seg, int(seg * hz) + 1)
        for t in times1:
            pt = JointTrajectoryPoint()
            pt.positions     = pchip1(t).tolist()
            pt.velocities    = pchip1.derivative(1)(t).tolist()
            pt.accelerations = pchip1.derivative(2)(t).tolist()
            pt.time_from_start.sec = int(t)
            pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
            traj.points.append(pt)

        # Sample segment 2 (skip first to avoid duplicate at 2.0s)
        times2 = np.linspace(0.0, seg, int(seg * hz) + 1)[1:]
        for tloc in times2:
            t = tloc + seg
            pt = JointTrajectoryPoint()
            pt.positions     = pchip2(tloc).tolist()
            pt.velocities    = pchip2.derivative(1)(tloc).tolist()
            pt.accelerations = pchip2.derivative(2)(tloc).tolist()
            pt.time_from_start.sec = int(t)
            pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
            traj.points.append(pt)

        self.traj_pub.publish(traj)
        self.get_logger().info(f"Published PCHIP cubic joint trajectory with {len(traj.points)} points.")
        # --- Quick JSSI plot (positions vs time) ---
        tsec = [p.time_from_start.sec + p.time_from_start.nanosec*1e-9 for p in traj.points]
        qmat = np.array([p.positions for p in traj.points])

        for i, name in enumerate(self.joint_names):
            plt.plot(tsec, qmat[:, i], label=name)

        plt.xlabel('Time (s)')
        plt.ylabel('Joint Position (rad)')
        plt.title('JSSI Joint Trajectory (PCHIP)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('jssi_joint_trajectory.png')
        plt.close()
        # -------------------------------------------

def main():
    rclpy.init()
    node = IKAndTrajectoryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
