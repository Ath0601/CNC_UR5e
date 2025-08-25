import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import numpy as np

class HomePosPublisher(Node):
    def __init__(self):
        super().__init__('home_pos_publisher')
        self.publisher_ = self.create_publisher(
            JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.current_positions = None
        self.got_joint_state = False
        self.done = False

        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        self.timer = self.create_timer(1.0, self.timer_callback)

    def joint_state_callback(self, msg):
        # Ensure we only get our robot's joints, in the right order!
        pos_dict = dict(zip(msg.name, msg.position))
        try:
            self.current_positions = np.array([pos_dict[name] for name in self.joint_names])
            self.got_joint_state = True
        except KeyError as e:
            self.get_logger().warn(f"Joint {e} not found in joint_states message.")

    def timer_callback(self):
        if self.done or not self.got_joint_state:
            if not self.got_joint_state:
                self.get_logger().info("Waiting for /joint_states message with correct joints...")
            return

        start = self.current_positions
        goal = np.array([0, -np.pi/2, (2*np.pi)/3, 0, np.pi/2, 0])
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = goal.tolist()
        point.velocities = [0.0]*6
        point.time_from_start.sec = 4
        point.time_from_start.nanosec = 0
        traj.points.append(point)

        self.publisher_.publish(traj)
        self.get_logger().info("Published home trajectory!")
        self.done = True

        if self.done and np.allclose(self.current_positions, goal, atol=0.03):
            self.get_logger().info("Reached home position, shutting down.")
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = HomePosPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
