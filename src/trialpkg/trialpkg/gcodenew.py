#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import csv
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from geometry_msgs.msg import PoseArray, Wrench
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.srv import GetCartesianPath

# --- ADDED ---
from std_msgs.msg import Float64, Float64MultiArray
# -------------

class Kalman1D:
    """Random-walk x_k = x_{k-1} + w; z_k = x_k + v."""
    def __init__(self, q=5.0, r=50.0, x0=None):
        self.q = float(q)   # process noise variance
        self.r = float(r)   # measurement noise variance
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

        # State
        self.last_joint_state = None
        self.gcode_waypoints = []
        self.gcode_received = False

        # Logging
        self.recording_active = False
        self.joint_efforts = []
        self.joint_velocities = []
        self.ft_data = []
        self.kf_force_q, self.kf_force_r = 5.0, 200.0
        self.kf_torque_q, self.kf_torque_r = 0.5, 10.0
        self._filt_fx = Kalman1D(self.kf_force_q,self.kf_force_r)
        self._filt_fy = Kalman1D(self.kf_force_q,self.kf_force_r)
        self._filt_fz = Kalman1D(self.kf_force_q,self.kf_force_r)
        self._filt_tx = Kalman1D(self.kf_torque_q,self.kf_torque_r)
        self._filt_ty = Kalman1D(self.kf_torque_q,self.kf_torque_r)
        self._filt_tz = Kalman1D(self.kf_torque_q,self.kf_torque_r)

        # Optional: also log a raw CSV for debugging
        self.ft_data_raw = []

        # Subs
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, qos)
        self.create_subscription(PoseArray, '/gcode_poses_base', self.gcode_cb, 10)
        self.create_subscription(Wrench, '/ft_sensor_wrench', self.ft_cb, 100)

        # Try start timer
        self.start_timer = self.create_timer(0.5, self.try_start)

        # -------- ADDED: spindle speed control ----------
        self.spindle_rpm_cmd = 1500.0      # default RPM; can be changed live
        self.spindle_running = False
        self.spindle_timer = None

        # publish velocity (rad/s) to JointGroupVelocityController
        self.spindle_pub = self.create_publisher(
            Float64MultiArray, '/spindle_velocity_controller/commands', 10
        )
        # accept live RPM updates on /spindle_rpm_cmd
        self.create_subscription(Float64, '/spindle_rpm_cmd', self._rpm_cb, 10)
        # ------------------------------------------------

    # ------------------- ADDED -------------------
    def _rpm_cb(self, msg: Float64):
        self.spindle_rpm_cmd = float(msg.data)
        self.get_logger().info(f"Spindle RPM command updated to {self.spindle_rpm_cmd:.0f}")

    def _start_spindle(self):
        if self.spindle_running:
            return
        self.spindle_running = True
        omega = self.spindle_rpm_cmd * 2.0 * math.pi / 60.0  # rad/s

        def _pub():
            if not self.spindle_running:
                return
            m = Float64MultiArray()
            m.data = [omega]  # controller expects one value for the spindle joint
            self.spindle_pub.publish(m)

        # publish at 50 Hz while running
        self.spindle_timer = self.create_timer(0.02, _pub)
        self.get_logger().info(f"Spindle started at {self.spindle_rpm_cmd:.0f} RPM ({omega:.2f} rad/s)")

    def _stop_spindle(self):
        if not self.spindle_running:
            return
        self.spindle_running = False
        if self.spindle_timer:
            self.spindle_timer.cancel()
            self.spindle_timer = None
        # send one explicit stop
        stop = Float64MultiArray()
        stop.data = [0.0]
        self.spindle_pub.publish(stop)
        self.get_logger().info("Spindle stopped.")
    # --------------------------------------------

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

        # Raw values
        fx, fy, fz = float(msg.force.x), float(msg.force.y), float(msg.force.z)
        tx, ty, tz = float(msg.torque.x), float(msg.torque.y), float(msg.torque.z)

        # Store raw (optional, for offline checks)
        self.ft_data_raw.append([now, fx, fy, fz, tx, ty, tz])
        Fx = self._filt_fx.update(fx)
        Fy = self._filt_fy.update(fy)
        Fz = self._filt_fz.update(fz)
        Tx = self._filt_tx.update(tx)
        Ty = self._filt_ty.update(ty)
        Tz = self._filt_tz.update(tz)

        # Log the FILTERED values for your plots/CSV
        self.ft_data.append([now, Fx, Fy, Fz, Tx, Ty, Tz])


    def gcode_cb(self, msg: PoseArray):
        self.gcode_waypoints = list(msg.poses)
        self.gcode_received = True
        self.get_logger().info(f"Received {len(self.gcode_waypoints)} G-code waypoints.")

    def try_start(self):
        if self.last_joint_state and self.gcode_received:
            self.start_timer.cancel()
            # ---- ADDED: start spindle just before executing path ----
            self._start_spindle()
            # ---------------------------------------------------------
            self.plan_and_execute_cartesian_async()

    def plan_and_execute_cartesian_async(self):
        if not self.gcode_waypoints:
            self.get_logger().error("No waypoints to plan.")
            return

        req = GetCartesianPath.Request()
        req.group_name = "manipulator"
        req.header.frame_id = "base_link"
        req.waypoints = self.gcode_waypoints
        req.max_step = 0.005  # 5mm resolution
        req.jump_threshold = 0.0
        req.avoid_collisions = True

        # Set start state to current joint state
        req.start_state.joint_state = self.last_joint_state

        future = self.cartesian_client.call_async(req)
        future.add_done_callback(self._after_cartesian_path)

    def _after_cartesian_path(self, fut):
        res = fut.result()
        if res is None or not res.solution.joint_trajectory.points:
            self.get_logger().error("Cartesian path planning failed or returned empty trajectory.")
            return

        self.get_logger().info(f"Planned Cartesian path fraction: {res.fraction*100:.1f}%")

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
            # ---- ADDED ----
            self._stop_spindle()
            # --------------
            return
        self.get_logger().info("G-code goal accepted; waiting for result...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._after_exec_finished)

    def _after_exec_finished(self, fut):
        self.recording_active = False
        self.get_logger().info("G-code execution finished.")
        # ---- ADDED: stop spindle after run ----
        self._stop_spindle()
        # ---------------------------------------
        self._write_csv_and_plots()
        self.destroy_node()

    def _write_csv_and_plots(self):
        if self.last_joint_state is None:
            return
        names = self.last_joint_state.name

        # CSVs
        with open('ft_sensor.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['time', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz'])
            w.writerows(self.ft_data)

        with open('joint_velocities.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['time'] + names)
            w.writerows(self.joint_velocities)

        # ------- CHANGED: separate force and torque plots -------
        # ---------- PLOTTING ----------
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

            # --- Force components ---
            plt.figure(figsize=(12, 6))
            plt.plot(t, fx, label='Fx [N]', color='r')
            plt.plot(t, fy, label='Fy [N]', color='g')
            plt.plot(t, fz, label='Fz [N]', color='b')
            ax = plt.gca()
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            plt.xlabel('Time [s]')
            plt.ylabel('Force [N]')
            plt.title('Forces over Time')
            plt.legend(loc='best')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig('forcecomp_plot_new.png')

            # --- Force magnitude ---
            plt.figure(figsize=(12, 6))
            plt.plot(t, force_mag, label='|F| [N]', color='k')
            ax = plt.gca()
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            plt.xlabel('Time [s]')
            plt.ylabel('Force Magnitude [N]')
            plt.title('Total Force Magnitude over Time')
            plt.legend(loc='best')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig('forcemag_plot_new.png')

            # --- Torque components ---
            plt.figure(figsize=(12, 6))
            plt.plot(t, tx, label='Tx [Nm]', color='r')
            plt.plot(t, ty, label='Ty [Nm]', color='g')
            plt.plot(t, tz, label='Tz [Nm]', color='b')
            ax = plt.gca()
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            plt.xlabel('Time [s]')
            plt.ylabel('Torque [Nm]')
            plt.title('Torques over Time')
            plt.legend(loc='best')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig('torquecomp_plot_new.png')

            # --- Torque magnitude ---
            plt.figure(figsize=(12, 6))
            plt.plot(t, torque_mag, label='|T| [Nm]', color='k')
            ax = plt.gca()
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            plt.xlabel('Time [s]')
            plt.ylabel('Torque Magnitude [Nm]')
            plt.title('Total Torque Magnitude over Time')
            plt.legend(loc='best')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig('torquemag_plot_new.png')

            # --- Spindle power (uses commanded ω; swap in measured if you have it) ---
            omega_cmd = self.spindle_rpm_cmd * 2.0 * math.pi / 60.0
            power_W   = [ty[i] * omega_cmd for i in range(len(t))]

            plt.figure(figsize=(12, 6))
            plt.plot(t, power_W, label='Power (W)', color='m')
            ax = plt.gca()
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            plt.xlabel('Time [s]')
            plt.ylabel('Power [W]')
            plt.title('Spindle Power vs Time (using commanded ω)')
            plt.legend(loc='best')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig('spindle_power_plot_new.png')

        # --- Joint velocities: split spindle vs other joints ---
        if self.joint_velocities and self.last_joint_state:
            names = self.last_joint_state.name
            t_j = [row[0] for row in self.joint_velocities]

            # Build per-joint series
            vel_by_joint = {name: [row[i+1] for row in self.joint_velocities] for i, name in enumerate(names)}

            # Plot robot joints (exclude end_mill_joint)
            plt.figure(figsize=(12, 6))
            for name in names:
                if name != 'end_mill_joint':
                    plt.plot(t_j, vel_by_joint[name], label=name)
            ax = plt.gca()
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            plt.xlabel('Time [s]')
            plt.ylabel('Velocity [rad/s]')
            plt.title('Joint Velocity vs Time (Robot joints only)')
            plt.legend(loc='best')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig('joint_velocities_robot_plot_new.png')

            # Plot end_mill_joint alone
            if 'end_mill_joint' in vel_by_joint:
                plt.figure(figsize=(12, 6))
                plt.plot(t_j, vel_by_joint['end_mill_joint'], label='end_mill_joint', color='purple')
                ax = plt.gca()
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                plt.xlabel('Time [s]')
                plt.ylabel('Velocity [rad/s]')
                plt.title('Spindle Joint Velocity (end_mill_joint)')
                plt.legend(loc='best')
                plt.grid(True, linestyle='--', alpha=0.3)
                plt.tight_layout()
                plt.savefig('joint_velocity_spindle_plot_new.png')
        # ---------- END PLOTTING ----------


def main():
    rclpy.init()
    node = GcodeExecNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
