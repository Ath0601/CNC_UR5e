#!/usr/bin/env python3
import csv
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from geometry_msgs.msg import PoseArray, Pose, Wrench
from sensor_msgs.msg import JointState
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.srv import GetCartesianPath
from std_msgs.msg import Float64, Float64MultiArray


# ----------------------- Geometry helpers -----------------------

def _pose_dist(a: Pose, b: Pose) -> float:
    return math.sqrt(
        (a.position.x - b.position.x) ** 2 +
        (a.position.y - b.position.y) ** 2 +
        (a.position.z - b.position.z) ** 2
    )

def _dedup_poses(poses, tol=1e-6):
    if not poses:
        return []
    out = [poses[0]]
    for p in poses[1:]:
        if _pose_dist(p, out[-1]) > tol:
            out.append(p)
    return out

def _cumulative_length(poses):
    if not poses:
        return [0.0]
    cum = [0.0]
    acc = 0.0
    for i in range(1, len(poses)):
        acc += _pose_dist(poses[i-1], poses[i])
        cum.append(acc)
    return cum


# ----------------- “L-shape” projection (constant Z pass) -----------------

def make_L_path_from_waypoints(waypoints, dx=0.002, z_tol=1e-6):
    """
    Build an L-shaped Cartesian path from arbitrary waypoints:
      - Keep the first safe waypoint (top of L).
      - Detect first plunge: first index where Z decreases by more than z_tol.
      - Fix target_z = Z at end of plunge segment.
      - For all subsequent waypoints, keep their X,Y but clamp Z = target_z.
      - Densify the long pass along XY with spacing dx.

    This produces |_____ with the bottom at constant Z, derived from the input,
    not hardcoded coordinates.
    """
    n = len(waypoints)
    if n <= 1:
        return waypoints[:]

    # 1) find first plunge (first negative ΔZ beyond tolerance)
    i_plunge = None
    for i in range(1, n):
        dz = waypoints[i].position.z - waypoints[i-1].position.z
        if dz < -z_tol:
            i_plunge = i
            break
    if i_plunge is None:
        # No plunge detected -> just return original
        return _dedup_poses(waypoints[:], tol=1e-8)

    # 2) lock Z to the depth reached at the end of the plunge
    target_z = waypoints[i_plunge].position.z

    # 3) Build dense XY path with Z locked
    dense = []

    # Top of L: keep original safe approach up to the plunge index
    dense.extend(waypoints[:i_plunge])

    # Add the plunge endpoint (the corner), ensure it uses target_z
    corner = Pose()
    corner.position.x = waypoints[i_plunge].position.x
    corner.position.y = waypoints[i_plunge].position.y
    corner.position.z = target_z
    corner.orientation = waypoints[i_plunge].orientation
    if not dense or _pose_dist(dense[-1], corner) > 1e-9:
        dense.append(corner)

    # Now sweep along the remaining XY with Z clamped
    # We’ll create a polyline of XY segments and densify each
    def _add_locked(pose: Pose):
        q = Pose()
        q.position.x = pose.position.x
        q.position.y = pose.position.y
        q.position.z = target_z
        q.orientation = pose.orientation
        return q

    prev = corner
    for k in range(i_plunge + 1, n):
        seg_end = _add_locked(waypoints[k])

        # densify from prev -> seg_end in XY
        dx_seg = seg_end.position.x - prev.position.x
        dy_seg = seg_end.position.y - prev.position.y
        seg_len = math.hypot(dx_seg, dy_seg)
        if seg_len < 1e-12:
            # same XY; just push the end
            if _pose_dist(prev, seg_end) > 1e-9:
                dense.append(seg_end)
            prev = seg_end
            continue

        steps = max(1, int(seg_len / dx))
        for s in range(1, steps + 1):
            a = s / steps
            p = Pose()
            p.position.x = prev.position.x + a * dx_seg
            p.position.y = prev.position.y + a * dy_seg
            p.position.z = target_z
            p.orientation = seg_end.orientation
            dense.append(p)
        prev = seg_end

    return _dedup_poses(dense, tol=1e-8)


# ----------------- Kalman (unchanged) ------------------

class Kalman1D:
    """Random-walk x_k = x_{k-1} + w; z_k = x_k + v."""
    def __init__(self, q=5.0, r=50.0, x0=None):
        self.q = float(q)
        self.r = float(r)
        self.x = float(x0) if x0 is not None else None
        self.P = 1e3 if x0 is None else 1.0

    def update(self, z):
        z = float(z)
        if self.x is None:
            self.x = z
            self.P = self.r
            return self.x
        x_pred = self.x
        P_pred = self.P + self.q
        K = P_pred / (P_pred + self.r)
        self.x = x_pred + K * (z - x_pred)
        self.P = (1.0 - K) * P_pred
        return self.x


# --------------------- Main Node -----------------------

class GcodeExecNode(Node):
    def __init__(self):
        super().__init__('gcode_exec_node')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )

        # Clients
        self.exec_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self.cartesian_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')

        self.get_logger().info("Waiting for /compute_cartesian_path service...")
        self.cartesian_client.wait_for_service()
        self.get_logger().info("Cartesian path service is available.")

        self.get_logger().info("Waiting for execute_trajectory action server...")
        self.exec_client.wait_for_server()
        self.get_logger().info("MoveIt execution action is available.")

        # Feed rate (mm/min) from coord node
        self.feed_mm_min = 300.0
        self.create_subscription(Float64, '/gcode_feed_rate', self._feed_cb, qos)

        # State
        self.last_joint_state = None
        self.gcode_waypoints = []
        self.gcode_received = True  # will flip true on callback

        # Logging/filters
        self.recording_active = False
        self.joint_efforts = []
        self.joint_velocities = []
        self.ft_data = []
        self.kf_force_q, self.kf_force_r = 5.0, 200.0
        self.kf_torque_q, self.kf_torque_r = 0.5, 10.0
        self._filt_fx = Kalman1D(self.kf_force_q, self.kf_force_r)
        self._filt_fy = Kalman1D(self.kf_force_q, self.kf_force_r)
        self._filt_fz = Kalman1D(self.kf_force_q, self.kf_force_r)
        self._filt_tx = Kalman1D(self.kf_torque_q, self.kf_torque_r)
        self._filt_ty = Kalman1D(self.kf_torque_q, self.kf_torque_r)
        self._filt_tz = Kalman1D(self.kf_torque_q, self.kf_torque_r)
        self.ft_data_raw = []

        # Subs
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, qos)
        self.create_subscription(PoseArray, '/gcode_poses_base', self.gcode_cb, 10)
        self.create_subscription(Wrench, '/ft_sensor_wrench', self.ft_cb, 100)

        # Spindle control
        self.spindle_rpm_cmd = 1500.0
        self.spindle_running = False
        self.spindle_timer = None
        self.spindle_pub = self.create_publisher(Float64MultiArray, '/spindle_velocity_controller/commands', 10)
        self.create_subscription(Float64, '/spindle_rpm_cmd', self._rpm_cb, 10)

        # Kickoff
        self.start_timer = self.create_timer(2.0, self.try_start)

    # ---------------- Callbacks ----------------

    def _feed_cb(self, msg: Float64):
        self.feed_mm_min = max(0.0, float(msg.data))
        self.get_logger().info(f"Feed override received: {self.feed_mm_min:.3f} mm/min")

    def _rpm_cb(self, msg: Float64):
        self.spindle_rpm_cmd = float(msg.data)
        self.get_logger().info(f"Spindle RPM command updated to {self.spindle_rpm_cmd:.0f}")

    def joint_state_cb(self, msg: JointState):
        self.last_joint_state = msg
        if self.recording_active:
            now = self.get_clock().now().nanoseconds * 1e-9
            self.joint_efforts.append([now] + list(msg.effort))
            self.joint_velocities.append([now] + list(msg.velocity))

    def ft_cb(self, msg: Wrench):
        if not self.recording_active:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        fx, fy, fz = float(msg.force.x), float(msg.force.y), float(msg.force.z)
        tx, ty, tz = float(msg.torque.x), float(msg.torque.y), float(msg.torque.z)
        self.ft_data_raw.append([now, fx, fy, fz, tx, ty, tz])
        Fx = self._filt_fx.update(fx)
        Fy = self._filt_fy.update(fy)
        Fz = self._filt_fz.update(fz)
        Tx = self._filt_tx.update(tx)
        Ty = self._filt_ty.update(ty)
        Tz = self._filt_tz.update(tz)
        self.ft_data.append([now, Fx, Fy, Fz, Tx, Ty, Tz])

    def gcode_cb(self, msg: PoseArray):
        self.gcode_waypoints = list(msg.poses)
        self.gcode_received = True
        try:
            if msg.header.frame_id:
                parsed = float(msg.header.frame_id)
                if parsed > 0.0:
                    self.feed_mm_min = parsed
                    self.get_logger().info(f"Feed from header.frame_id: {self.feed_mm_min:.3f} mm/min")
        except Exception:
            pass
        self.get_logger().info(f"Received {len(self.gcode_waypoints)} G-code waypoints. "
                               f"Current feed: {self.feed_mm_min:.3f} mm/min.")

    def try_start(self):
        if self.last_joint_state and self.gcode_received:
            self.start_timer.cancel()
            self._start_spindle()
            self.plan_and_execute_cartesian_async()

    # ---------------- Spindle control ----------------

    def _start_spindle(self):
        """Start/keep spindle at commanded RPM with a small keepalive timer."""
        if self.spindle_running:
            return
        self.spindle_running = True
        # publish immediately
        self._keep_spindle()
        # keepalive at 5 Hz so Gazebo controller doesn't time out
        self.spindle_timer = self.create_timer(0.2, self._keep_spindle)
        self.get_logger().info(f"Spindle started at ~{self.spindle_rpm_cmd:.0f} RPM")

    def _keep_spindle(self):
        msg = Float64MultiArray()
        msg.data = [float(self.spindle_rpm_cmd)]
        self.spindle_pub.publish(msg)

    def _stop_spindle(self):
        if not self.spindle_running:
            return
        if self.spindle_timer is not None:
            self.spindle_timer.cancel()
            self.spindle_timer = None
        msg = Float64MultiArray()
        msg.data = [0.0]
        self.spindle_pub.publish(msg)
        self.spindle_running = False
        self.get_logger().info("Spindle stopped")

    # ---------------- Planning/Execution ----------------

    def plan_and_execute_cartesian_async(self):
        if not self.gcode_waypoints:
            self.get_logger().error("No waypoints to plan.")
            return

        # Build a true “L” path with constant Z pass (derived, not hardcoded)
        engaged_wps = make_L_path_from_waypoints(self.gcode_waypoints, dx=0.002)
        engaged_wps = _dedup_poses(engaged_wps, tol=1e-8)

        req = GetCartesianPath.Request()
        req.group_name = "manipulator"
        req.header.frame_id = "base_link"
        req.waypoints = engaged_wps
        req.max_step = 0.002
        req.jump_threshold = 0.0
        req.avoid_collisions = False  # keep contact with stock
        req.start_state.joint_state = self.last_joint_state

        future = self.cartesian_client.call_async(req)
        future.add_done_callback(lambda fut: self._after_cartesian_path(fut, engaged_wps))

    def _after_cartesian_path(self, fut, engaged_wps):
        res = fut.result()
        if res is None or not res.solution.joint_trajectory.points:
            self.get_logger().error("Cartesian path planning failed or returned empty trajectory.")
            return

        self.get_logger().info(f"Planned Cartesian path fraction: {res.fraction*100:.1f}%")
        traj = res.solution.joint_trajectory
        points = traj.points
        Nt = len(points)

        # ---- FEED-ACCURATE, STRICTLY MONOTONIC RETIMING ----
        cum = _cumulative_length(engaged_wps)
        total_len = cum[-1] if cum else 0.0
        feed_m_s = max(1e-9, (self.feed_mm_min / 1000.0) / 60.0)
        total_time = (total_len / feed_m_s) if total_len > 0 else 0.0

        min_dt = 1e-4  # enforce strictly increasing

        if Nt == 1 or total_time == 0.0:
            points[0].time_from_start.sec = 0
            points[0].time_from_start.nanosec = int(min_dt * 1e9)
        else:
            Nw = len(engaged_wps)
            last_t = -1.0
            for i in range(Nt):
                alpha = i / (Nt - 1)
                w_float = alpha * (Nw - 1)
                w0 = int(math.floor(w_float))
                w1 = min(w0 + 1, Nw - 1)
                frac = w_float - w0
                d = (1.0 - frac) * cum[w0] + frac * cum[w1]
                t_i = total_time * (d / total_len if total_len > 0 else 0.0)
                if t_i <= last_t:
                    t_i = last_t + min_dt
                last_t = t_i
                sec = int(t_i)
                nsec = int((t_i - sec) * 1e9)
                points[i].time_from_start.sec = sec
                points[i].time_from_start.nanosec = nsec

        res.solution.joint_trajectory.points = points

        # Execute
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = res.solution

        self.recording_active = True
        send_future = self.exec_client.send_goal_async(goal)
        send_future.add_done_callback(self._after_exec_goal_sent)

    def _after_exec_goal_sent(self, fut):
        goal_handle = fut.result()
        if not goal_handle.accepted:
            self.get_logger().error("G-code trajectory goal rejected.")
            self.recording_active = False
            self._stop_spindle()
            return
        self.get_logger().info("G-code goal accepted; waiting for result...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._after_exec_finished)

    def _after_exec_finished(self, fut):
        self.recording_active = False
        self.get_logger().info("G-code execution finished.")
        self._stop_spindle()
        self._write_csv_and_plots()
        self.destroy_node()

    # ----------------- CSV + Plots -----------------

    def _write_csv_and_plots(self):
        if self.last_joint_state is None:
            return
        names = self.last_joint_state.name

        with open('ft_sensor.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['time', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz'])
            w.writerows(self.ft_data)

        with open('joint_velocities.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['time'] + names)
            w.writerows(self.joint_velocities)

        if self.ft_data:
            t  = [row[0] for row in self.ft_data]
            fx = [row[1] for row in self.ft_data]
            fy = [row[2] for row in self.ft_data]
            fz = [row[3] for row in self.ft_data]
            tx = [row[4] for row in self.ft_data]
            ty = [row[5] for row in self.ft_data]
            tz = [row[6] for row in self.ft_data]

            force_mag  = [(fx[i]**2 + fy[i]**2 + fz[i]**2)**0.5 for i in range(len(t))]
            torque_mag = [(tx[i]**2 + ty[i]**2 + tz[i]**2)**0.5 for i in range(len(t))]

            plt.figure(figsize=(12, 6))
            plt.plot(t, fx, label='Fx [N]', color='r')
            plt.plot(t, fy, label='Fy [N]', color='g')
            plt.plot(t, fz, label='Fz [N]', color='b')
            plt.xlabel('Time [s]'); plt.ylabel('Force [N]')
            plt.title('Force Components over Time')
            plt.legend(loc='best'); plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout(); plt.savefig('force_comp_plot_path.png')

            plt.figure(figsize=(12, 6))
            plt.plot(t, force_mag, label='|F| [N]', color='k')
            plt.xlabel('Time [s]'); plt.ylabel('Force Magnitude [N]')
            plt.title('Total Force Magnitude over Time')
            plt.legend(loc='best'); plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout(); plt.savefig('force_mag_plot_path.png')

            plt.figure(figsize=(12, 6))
            plt.plot(t, tx, label='Tx [Nm]', color='r')
            plt.plot(t, ty, label='Ty [Nm]', color='g')
            plt.plot(t, tz, label='Tz [Nm]', color='b')
            plt.xlabel('Time [s]'); plt.ylabel('Torque [Nm]')
            plt.title('Torque Components over Time')
            plt.legend(loc='best'); plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout(); plt.savefig('torque_comp_plot_path.png')

            plt.figure(figsize=(12, 6))
            plt.plot(t, torque_mag, label='|T| [Nm]', color='k')
            plt.xlabel('Time [s]'); plt.ylabel('Torque Magnitude [Nm]')
            plt.title('Total Torque Magnitude over Time')
            plt.legend(loc='best'); plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout(); plt.savefig('torque_mag_plot_path.png')

            omega_cmd = self.spindle_rpm_cmd * 2.0 * math.pi / 60.0
            power_W   = [ty[i] * omega_cmd for i in range(len(t))]
            plt.figure(figsize=(12, 6))
            plt.plot(t, power_W, label='Power (W)', color='m')
            ax = plt.gca(); ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            plt.xlabel('Time [s]'); plt.ylabel('Power [W]')
            plt.title('Spindle Power vs Time (using commanded ω)')
            plt.legend(loc='best'); plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout(); plt.savefig('spindle_p_plot_path.png')


def main():
    rclpy.init()
    node = GcodeExecNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
