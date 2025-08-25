#!/usr/bin/env python3
"""
Straight-line Cartesian executor with:
- Fixed tool-down orientation (constant quaternion)
- TCP straightness monitor (CSV + XY/XZ/3D plots + lateral deviation vs time)
- Force/Torque logging with 1D Kalman filtering (Fx,Fy,Fz, Tx,Ty,Tz)
- Spindle power plot (P = Tz * omega_cmd)
- Spindle keepalive publisher (velocity command)
- Feed-rate subscription (/gcode_feed_rate), CSV, plot
- Linear cutting power using feed rate: P_linear = F_parallel * v_feed

Outputs after each run:
- executed_tcp_path.csv
- straightness_report.txt
- tcp_path_xy.png
- tcp_path_xz.png
- lateral_deviation_vs_time.png
- tcp_path_3d.png (if mpl3d available)
- ft_sensor.csv
- joint_velocities.csv
- force_components_plot.png
- force_magnitude_plot.png
- torque_components_plot.png
- torque_magnitude_plot.png
- spindle_power_plot.png
- feed_rate.csv
- feed_rate_plot.png
- cutting_power_linear_plot.png
"""

import math
import csv
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import Pose, PoseArray, TransformStamped, Wrench
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import MoveItErrorCodes

import tf2_ros
from tf2_ros import TransformException

# --- plotting (headless) ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from std_msgs.msg import Float64, Float64MultiArray


# ----------------- Small helpers -----------------

def set_quat(p: Pose, qx: float, qy: float, qz: float, qw: float) -> None:
    p.orientation.x = qx
    p.orientation.y = qy
    p.orientation.z = qz
    p.orientation.w = qw


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def straight_resample(p0: Pose, p1: Pose, eef_step: float = 0.002) -> List[Pose]:
    dx = p1.position.x - p0.position.x
    dy = p1.position.y - p0.position.y
    dz = p1.position.z - p0.position.z
    L = math.sqrt(dx*dx + dy*dy + dz*dz)
    if L < 1e-9:
        return [p0]
    n = max(2, int(math.ceil(L / max(1e-6, eef_step))) + 1)
    out: List[Pose] = []
    for i in range(n):
        t = i / (n - 1)
        pi = Pose()
        pi.position.x = lerp(p0.position.x, p1.position.x, t)
        pi.position.y = lerp(p0.position.y, p1.position.y, t)
        pi.position.z = lerp(p0.position.z, p1.position.z, t)
        out.append(pi)
    return out


def dedup_poses(poses: List[Pose], tol: float = 1e-12) -> List[Pose]:
    if not poses:
        return []
    out = [poses[0]]
    last = poses[0]
    for p in poses[1:]:
        dx = p.position.x - last.position.x
        dy = p.position.y - last.position.y
        dz = p.position.z - last.position.z
        if dx*dx + dy*dy + dz*dz > tol*tol:
            out.append(p)
            last = p
    return out


def point_to_line_distance(px, py, pz, ax, ay, az, bx, by, bz) -> float:
    """Perpendicular distance from point P to infinite line through A->B (3D)."""
    vx, vy, vz = bx - ax, by - ay, bz - az
    wx, wy, wz = px - ax, py - ay, pz - az
    v_norm = math.sqrt(vx*vx + vy*vy + vz*vz)
    if v_norm < 1e-12:
        return math.sqrt(wx*wx + wy*wy + wz*wz)
    cx = wy * vz - wz * vy
    cy = wz * vx - wx * vz
    cz = wx * vy - wy * vx
    cross_norm = math.sqrt(cx*cx + cy*cy + cz*cz)
    return cross_norm / v_norm


def project_point_on_line(px, py, pz, ax, ay, az, bx, by, bz):
    """Return param t in [0,1] for projection of P onto segment A->B (clamped), plus projected point."""
    vx, vy, vz = bx - ax, by - ay, bz - az
    wx, wy, wz = px - ax, py - ay, pz - az
    denom = vx*vx + vy*vy + vz*vz
    if denom < 1e-18:
        return 0.0, (ax, ay, az)
    t = (wx*vx + wy*vy + wz*vz) / denom
    qx = ax + t * vx
    qy = ay + t * vy
    qz = az + t * vz
    return t, (qx, qy, qz)


# ----------------- Simple 1D Kalman Filter -----------------

class Kalman1D:
    """Random-walk state: x_k = x_{k-1} + w, z_k = x_k + v."""
    def __init__(self, q=5.0, r=200.0, x0=None):
        self.q = float(q)   # process variance
        self.r = float(r)   # measurement variance
        self.x = float(x0) if x0 is not None else None
        self.P = 1e3 if x0 is None else 1.0

    def update(self, z):
        z = float(z)
        if self.x is None:
            self.x = z
            self.P = self.r
            return self.x
        # Predict
        x_pred = self.x
        P_pred = self.P + self.q
        # Update
        K = P_pred / (P_pred + self.r)
        self.x = x_pred + K * (z - x_pred)
        self.P = (1.0 - K) * P_pred
        return self.x


# ----------------- The Node -----------------

class StraightGcodeExecutor(Node):
    def __init__(self):
        super().__init__('straight_gcode_exec')

        # ---- CONFIG ----
        self.base_link = 'base_link'
        self.tip_link = 'tool0_link'   # <-- set to your real TCP-at-tip link
        # tool_down_quat = R.from_euler('xyz', [-pi/2, 0, 0]).as_quat()
        SQ2_2 = math.sqrt(0.5)
        self.tool_down_quat = (-SQ2_2, 0.0, 0.0, SQ2_2)  # (x, y, z, w)
        self.max_step = 0.002          # 2 mm planning resolution
        self.jump_threshold = 1.5       # enable jump detection
        self.require_fraction = 0.98    # require >= 98% completion
        self.sample_rate = 50.0         # Hz sampling of executed TCP via TF

        # Spindle & power
        self.spindle_rpm_cmd = 1500.0   # can be overridden via /spindle_rpm_cmd
        self.spindle_axis = 'y'         # use 'x'/'y' if torque axis differs

        # Feed rate (mm/min) from /gcode_feed_rate
        self.feed_rate_mm_min = 300.0   # default if topic not yet received
        self.feed_series = []           # [t, feed_mm_min]

        # Kalman filter tuning (forces/torques)
        self.kf_force_q, self.kf_force_r = 5.0, 200.0
        self.kf_torque_q, self.kf_torque_r = 0.5, 10.0
        # ----------------

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )

        # MoveIt service/action
        self.cartesian_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.exec_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')

        self.get_logger().info("Waiting for /compute_cartesian_path ...")
        self.cartesian_client.wait_for_service()
        self.get_logger().info("Waiting for /execute_trajectory ...")
        self.exec_client.wait_for_server()
        self.get_logger().info("Services ready.")

        # Subscriptions
        self.last_joint_state: Optional[JointState] = None
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, qos)
        self.create_subscription(PoseArray, '/gcode_poses_base', self.gcode_cb, 10)
        self.create_subscription(Wrench, '/ft_sensor_wrench', self.ft_cb, 100)
        self.create_subscription(Float64, '/gcode_feed_rate', self._feed_cb, 10)   # <-- feed rate
        self.create_subscription(Float64, '/spindle_rpm_cmd', self._rpm_cb, 10)    # optional RPM override

        # Spindle publisher (keepalive)
        self.spindle_pub = self.create_publisher(Float64MultiArray, '/spindle_velocity_controller/commands', 10)
        self.spindle_running = False
        self.spindle_timer = None

        # G-code storage
        self.gcode_waypoints: List[Pose] = []
        self.have_waypoints = False

        # Straightness monitor storage
        self.ideal_a = None  # (ax, ay, az)
        self.ideal_b = None  # (bx, by, bz)
        self.path_samples = []  # [t, x, y, z]
        self.exec_sampling_timer = None
        self.exec_active = False

        # TF for executed TCP sampling
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)

        # Force/torque logging (filtered + raw)
        self.recording_active = False
        self.ft_data = []      # [t, Fx,Fy,Fz, Tx,Ty,Tz] (filtered)
        self.ft_data_raw = []  # [t, fx,fy,fz, tx,ty,tz] (raw)
        self.joint_velocities = []  # [t, vel...]

        # filters per axis
        self._filt_fx = Kalman1D(self.kf_force_q, self.kf_force_r)
        self._filt_fy = Kalman1D(self.kf_force_q, self.kf_force_r)
        self._filt_fz = Kalman1D(self.kf_force_q, self.kf_force_r)
        self._filt_tx = Kalman1D(self.kf_torque_q, self.kf_torque_r)
        self._filt_ty = Kalman1D(self.kf_torque_q, self.kf_torque_r)
        self._filt_tz = Kalman1D(self.kf_torque_q, self.kf_torque_r)

        # Kickoff when ready
        self.start_timer = self.create_timer(0.2, self.try_start)

    # ------------- Callbacks -------------

    def joint_state_cb(self, msg: JointState):
        self.last_joint_state = msg
        if self.recording_active:
            now = self.get_clock().now().nanoseconds * 1e-9
            self.joint_velocities.append([now] + list(msg.velocity))

    def gcode_cb(self, msg: PoseArray):
        self.gcode_waypoints = list(msg.poses)
        self.have_waypoints = True
        self.get_logger().info(f"Received {len(self.gcode_waypoints)} G-code poses.")

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

    def _feed_cb(self, msg: Float64):
        self.feed_rate_mm_min = float(msg.data)
        if self.recording_active:
            now = self.get_clock().now().nanoseconds * 1e-9
            self.feed_series.append([now, self.feed_rate_mm_min])
        self.get_logger().info(f"Feed rate updated: {self.feed_rate_mm_min:.1f} mm/min")

    def _rpm_cb(self, msg: Float64):
        self.spindle_rpm_cmd = float(msg.data)
        self.get_logger().info(f"Spindle RPM command updated to {self.spindle_rpm_cmd:.0f}")

    def try_start(self):
        if self.exec_active:
            return
        if self.have_waypoints and self.last_joint_state is not None:
            self.start_timer.cancel()
            self.plan_and_execute_cartesian_async()

    # ------------- Planning/Execution -------------

    def plan_and_execute_cartesian_async(self):
        if len(self.gcode_waypoints) < 2:
            self.get_logger().error("Need at least two waypoints.")
            return

        src = dedup_poses(self.gcode_waypoints, tol=1e-9)
        if len(src) < 2:
            self.get_logger().error("Waypoints collapsed to <2 after dedup.")
            return
        p0, p1 = src[0], src[-1]

        # Build dense, perfectly straight segment
        straight = straight_resample(p0, p1, eef_step=self.max_step)

        # Force constant tool-down orientation
        qx, qy, qz, qw = self.tool_down_quat
        for p in straight:
            set_quat(p, qx, qy, qz, qw)

        # Save ideal line endpoints for straightness monitor (in base frame)
        self.ideal_a = (p0.position.x, p0.position.y, p0.position.z)
        self.ideal_b = (p1.position.x, p1.position.y, p1.position.z)

        # Compose request
        req = GetCartesianPath.Request()
        req.group_name = "manipulator"
        req.header.frame_id = self.base_link
        req.link_name = self.tip_link
        req.waypoints = straight
        req.max_step = self.max_step
        req.jump_threshold = self.jump_threshold
        req.avoid_collisions = False
        req.start_state.joint_state = self.last_joint_state

        self.get_logger().info(f"Planning straight line: {len(straight)} samples, step={self.max_step:.3f} m")
        future = self.cartesian_client.call_async(req)
        future.add_done_callback(self._after_cartesian_path)

    def _after_cartesian_path(self, fut):
        res = fut.result()
        if res is None or not res.solution.joint_trajectory.points:
            self.get_logger().error("Cartesian planning failed or empty trajectory.")
            return

        fraction = float(res.fraction)
        self.get_logger().info(f"Planned Cartesian path fraction: {fraction*100:.1f}%")
        if fraction < self.require_fraction:
            self.get_logger().error(f"Fraction {fraction:.3f} < require {self.require_fraction:.3f}; abort.")
            return

        # Execute
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = res.solution

        # Start spindle keepalive + logging
        self._start_spindle()
        self.recording_active = True
        # capture initial feed immediately
        self.feed_series.append([self.get_clock().now().nanoseconds * 1e-9, self.feed_rate_mm_min])

        self.exec_active = True
        send_future = self.exec_client.send_goal_async(goal)
        send_future.add_done_callback(self._after_exec_goal_sent)

    def _after_exec_goal_sent(self, fut):
        goal_handle = fut.result()
        if not goal_handle.accepted:
            self.get_logger().error("ExecuteTrajectory goal rejected.")
            self.exec_active = False
            self.recording_active = False
            self._stop_spindle()
            return

        self.get_logger().info("ExecuteTrajectory accepted. Sampling executed TCP path...")
        # Start sampling executed TCP pose via TF at fixed rate
        period = 1.0 / self.sample_rate
        if self.exec_sampling_timer is None:
            self.exec_sampling_timer = self.create_timer(period, self._sample_tcp_pose)

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._after_exec_finished)

    def _sample_tcp_pose(self):
        # Collect base_link -> tip_link pose
        try:
            t: TransformStamped = self.tf_buffer.lookup_transform(
                self.base_link, self.tip_link, rclpy.time.Time())
            tx = t.transform.translation.x
            ty = t.transform.translation.y
            tz = t.transform.translation.z
            tn = self.get_clock().now().nanoseconds * 1e-9
            self.path_samples.append([tn, tx, ty, tz])
        except TransformException:
            pass

    def _after_exec_finished(self, fut):
        result = fut.result()
        code = result.result.error_code.val if result and result.result else None

        # Stop sampling & logging
        if self.exec_sampling_timer is not None:
            self.exec_sampling_timer.cancel()
            self.exec_sampling_timer = None
        self.exec_active = False
        self.recording_active = False
        self._stop_spindle()

        if code == MoveItErrorCodes.SUCCESS:
            self.get_logger().info("Execution SUCCESS.")
        else:
            self.get_logger().error(f"Execution ABORTED (MoveIt code {code}).")

        # Save outputs
        self._write_straightness_outputs()
        self._write_ft_csv_and_plots()
        self._write_feed_csv_and_plot()
        self.destroy_node()

    # ------------- Spindle keepalive -------------

    def _start_spindle(self):
        if self.spindle_running:
            return
        self.spindle_running = True
        self._keep_spindle()
        self.spindle_timer = self.create_timer(0.2, self._keep_spindle)  # 5 Hz keepalive
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

    # ------------- Straightness reporting & plots -------------

    def _write_straightness_outputs(self):
        # Write executed path CSV
        if self.path_samples:
            with open('executed_tcp_path.csv', 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['time_s', 'x', 'y', 'z'])
                w.writerows(self.path_samples)
            self.get_logger().info("Saved executed TCP path to executed_tcp_path.csv")
        else:
            self.get_logger().warn("No executed TCP samples captured.")

        # Compute straightness metrics
        if not (self.ideal_a and self.ideal_b and self.path_samples):
            return

        ax, ay, az = self.ideal_a
        bx, by, bz = self.ideal_b

        ts = [row[0] for row in self.path_samples]
        xs = [row[1] for row in self.path_samples]
        ys = [row[2] for row in self.path_samples]
        zs = [row[3] for row in self.path_samples]

        dists = []
        tpars = []     # param along the ideal line
        for x, y, z in zip(xs, ys, zs):
            d = point_to_line_distance(x, y, z, ax, ay, az, bx, by, bz)
            t_param, _ = project_point_on_line(x, y, z, ax, ay, az, bx, by, bz)
            dists.append(d)
            tpars.append(t_param)

        if not dists:
            return

        max_dev = max(dists)
        rms = math.sqrt(sum(d*d for d in dists) / len(dists))
        avg = sum(dists) / len(dists)

        with open('straightness_report.txt', 'w') as f:
            f.write("Straightness report (distances are perpendicular to ideal line)\n")
            f.write(f"Tip link: {self.tip_link}\n")
            f.write(f"Base link: {self.base_link}\n")
            f.write(f"Ideal line start (m): {self.ideal_a}\n")
            f.write(f"Ideal line end   (m): {self.ideal_b}\n")
            f.write(f"Samples: {len(dists)}\n")
            f.write(f"Max deviation (m): {max_dev:.6f}\n")
            f.write(f"RMS deviation (m): {rms:.6f}\n")
            f.write(f"Mean deviation (m): {avg:.6f}\n")

        self.get_logger().info(
            f"Straightness -> max: {max_dev*1000:.2f} mm, RMS: {rms*1000:.2f} mm "
            f"(report: straightness_report.txt)"
        )

        # Reconstruct ideal points corresponding to each sampled param t in [0,1]
        ix = [ax + tp * (bx - ax) for tp in tpars]
        iy = [ay + tp * (by - ay) for tp in tpars]
        iz = [az + tp * (bz - az) for tp in tpars]

        # XY plot
        plt.figure()
        plt.plot(ix, iy, label='Ideal line (XY)', linewidth=2)
        plt.plot(xs, ys, '.', label='Executed TCP (XY)', markersize=3)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('TCP Path: XY Projection')
        plt.legend(loc='best')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('tcp_path_xy.png')
        plt.close()

        # XZ plot
        plt.figure()
        plt.plot(ix, iz, label='Ideal line (XZ)', linewidth=2)
        plt.plot(xs, zs, '.', label='Executed TCP (XZ)', markersize=3)
        plt.xlabel('X (m)')
        plt.ylabel('Z (m)')
        plt.title('TCP Path: XZ Projection')
        plt.legend(loc='best')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('tcp_path_xz.png')
        plt.close()

        # Lateral deviation vs time
        plt.figure()
        plt.plot(ts, [d*1000.0 for d in dists], label='Lateral deviation (mm)')
        plt.xlabel('Time (s)')
        plt.ylabel('Deviation (mm)')
        plt.title('Lateral Deviation from Ideal Line vs Time')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig('lateral_deviation_vs_time.png')
        plt.close()

        # Optional 3D path (executed & ideal)
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig = plt.figure()
            ax3d = fig.add_subplot(111, projection='3d')
            ax3d.plot(ix, iy, iz, label='Ideal line', linewidth=2)
            ax3d.plot(xs, ys, zs, '.', label='Executed TCP', markersize=2)
            ax3d.set_xlabel('X (m)')
            ax3d.set_ylabel('Y (m)')
            ax3d.set_zlabel('Z (m)')
            ax3d.set_title('TCP Path: 3D View')
            ax3d.legend(loc='best')
            plt.tight_layout()
            plt.savefig('tcp_path_3d.png')
            plt.close()
        except Exception as e:
            self.get_logger().warn(f"3D plot skipped: {e}")

    # ------------- Force/Torque + Power CSV & Plots -------------

    def _write_ft_csv_and_plots(self):
        # CSVs
        with open('ft_sensor.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['time', 'Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz'])
            w.writerows(self.ft_data)

        if self.joint_velocities and self.last_joint_state is not None:
            names = self.last_joint_state.name
            with open('joint_velocities.csv', 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['time'] + names)
                w.writerows(self.joint_velocities)

        if not self.ft_data:
            self.get_logger().warn("No F/T samples captured.")
            return

        t  = [row[0] for row in self.ft_data]
        Fx = [row[1] for row in self.ft_data]
        Fy = [row[2] for row in self.ft_data]
        Fz = [row[3] for row in self.ft_data]
        Tx = [row[4] for row in self.ft_data]
        Ty = [row[5] for row in self.ft_data]
        Tz = [row[6] for row in self.ft_data]
        tv = [row[0] for row in self.joint_velocities]
        # Convert rad/s -> deg/s for readability
        rad2deg = 180.0 / math.pi

        force_mag  = [(Fx[i]**2 + Fy[i]**2 + Fz[i]**2)**0.5 for i in range(len(t))]
        torque_mag = [(Tx[i]**2 + Ty[i]**2 + Tz[i]**2)**0.5 for i in range(len(t))]

        # Force components
        plt.figure(figsize=(12, 6))
        plt.plot(t, Fx, label='Fx [N]')
        plt.plot(t, Fy, label='Fy [N]')
        plt.plot(t, Fz, label='Fz [N]')
        plt.xlabel('Time [s]'); plt.ylabel('Force [N]')
        plt.title('Force Components over Time')
        plt.legend(loc='best'); plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout(); plt.savefig('force_components_plot.png'); plt.close()

        # Force magnitude
        plt.figure(figsize=(12, 6))
        plt.plot(t, force_mag, label='|F| [N]')
        plt.xlabel('Time [s]'); plt.ylabel('Force [N]')
        plt.title('Total Force Magnitude over Time')
        plt.legend(loc='best'); plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout(); plt.savefig('force_magnitude_plot.png'); plt.close()

        # Torque components
        plt.figure(figsize=(12, 6))
        plt.plot(t, Tx, label='Tx [Nm]')
        plt.plot(t, Ty, label='Ty [Nm]')
        plt.plot(t, Tz, label='Tz [Nm]')
        plt.xlabel('Time [s]'); plt.ylabel('Torque [Nm]')
        plt.title('Torque Components over Time')
        plt.legend(loc='best'); plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout(); plt.savefig('torque_components_plot.png'); plt.close()

        # Torque magnitude
        plt.figure(figsize=(12, 6))
        plt.plot(t, torque_mag, label='|T| [Nm]')
        plt.xlabel('Time [s]'); plt.ylabel('Torque [Nm]')
        plt.title('Total Torque Magnitude over Time')
        plt.legend(loc='best'); plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout(); plt.savefig('torque_magnitude_plot.png'); plt.close()

        plt.figure(figsize=(12, 6))
        for j, name in enumerate(names):
            vals_deg = [row[j+1] * rad2deg for row in self.joint_velocities]  # j+1 because first column is time
            plt.plot(tv, vals_deg, label=name)
        plt.xlabel('Time [s]'); plt.ylabel('Velocity [deg/s]')
        plt.title('Joint Velocities vs Time')
        plt.legend(loc='best', ncol=2); plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout(); plt.savefig('joint_velocities_plot.png'); plt.close()

        omega_cmd = self.spindle_rpm_cmd * 2.0 * math.pi / 60.0  # rad/s
        power_rot_W = [Ty[i] * omega_cmd for i in range(len(t))]

        plt.figure(figsize=(12, 6))
        plt.plot(t, power_rot_W, label='Spindle Power (rotational, W)')
        plt.xlabel('Time [s]'); plt.ylabel('Power [W]')
        plt.title('Spindle Power vs Time (using commanded ω)')
        plt.legend(loc='best'); plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout(); plt.savefig('spindle_power_plot.png'); plt.close()

        # -------- Linear cutting power using FEED RATE --------
        # Unit direction of planned straight line (in base frame)
        if self.ideal_a and self.ideal_b:
            ax, ay, az = self.ideal_a
            bx, by, bz = self.ideal_b
            ux, uy, uz = (bx - ax), (by - ay), (bz - az)
            norm = math.sqrt(ux*ux + uy*uy + uz*uz)
            if norm > 1e-12:
                ux, uy, uz = ux / norm, uy / norm, uz / norm
            else:
                ux, uy, uz = 1.0, 0.0, 0.0  # fallback

            # Force component along path
            F_parallel = [Fx[i]*ux + Fy[i]*uy + Fz[i]*uz for i in range(len(t))]
        else:
            F_parallel = [0.0 for _ in t]

        v_feed_mps = float(self.feed_rate_mm_min) / 60000.0  # mm/min -> m/s
        power_linear_W = [F_parallel[i] * v_feed_mps for i in range(len(t))]

        plt.figure(figsize=(12, 6))
        plt.plot(t, power_linear_W, label='Cutting Power (linear, W)')
        plt.xlabel('Time [s]'); plt.ylabel('Power [W]')
        plt.title('Linear Cutting Power vs Time (F_parallel · v_feed)')
        plt.legend(loc='best'); plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout(); plt.savefig('cutting_power_linear_plot.png'); plt.close()
    
    # ------------- Feed-rate CSV & Plot -------------

    def _write_feed_csv_and_plot(self):
        # Append last value if we only captured at start
        if not self.feed_series:
            return
        with open('feed_rate.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['time', 'feed_mm_per_min'])
            w.writerows(self.feed_series)

        tt = [row[0] for row in self.feed_series]
        ff = [row[1] for row in self.feed_series]

        plt.figure(figsize=(10, 5))
        plt.step(tt, ff, where='post', label='Feed [mm/min]')
        plt.xlabel('Time [s]'); plt.ylabel('Feed [mm/min]')
        plt.title('G-code Feed Rate vs Time')
        plt.legend(loc='best'); plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout(); plt.savefig('feed_rate_plot.png'); plt.close()


# ----------------- Main -----------------

def main():
    rclpy.init()
    node = StraightGcodeExecutor()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
