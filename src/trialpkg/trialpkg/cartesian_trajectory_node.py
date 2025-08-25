import rclpy, numpy as np
from rclpy.node import Node
from moveit_msgs.srv import GetCartesianPath, GetStateValidity, GetPositionIK
from moveit_msgs.action import ExecuteTrajectory
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from scipy.spatial.transform import Rotation as R
from rclpy.action import ActionClient
from trialpkg.gcodeparsernew import GCodeParser

class CartesianPathClient(Node):
    def __init__(self):
        super().__init__('cartesian_path_client')
        self.cartesian_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self.marker_pub = self.create_publisher(MarkerArray, '/gcode_waypoint_markers', 1)
        self.posearray_pub = self.create_publisher(PoseArray, '/gcode_waypoint_poses', 1)
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.get_logger().info('Waiting for compute_cartesian_path service...')
        self.cartesian_client.wait_for_service()
        self.get_logger().info('Waiting for execute_trajectory action server...')
        self.execute_client.wait_for_server()
        self.state_validity_client = self.create_client(GetStateValidity, '/check_state_validity')
        self.state_validity_client.wait_for_service()

    def check_collision(self, joint_state, group='manipulator'):
        req = GetStateValidity.Request()
        req.robot_state.joint_state = joint_state
        req.group_name = group
        future = self.state_validity_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        return result.valid, result.contacts
    
    def visualize_waypoints(self, poses):
        marker_array = MarkerArray()
        for i, pose in enumerate(poses):
            m = Marker()
            m.header.frame_id = 'base_link'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'gcode_waypoints'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose = pose
            m.scale.x = m.scale.y = m.scale.z = 0.02
            m.color.r = 1.0
            m.color.g = 0.2
            m.color.b = 0.2
            m.color.a = 1.0
            marker_array.markers.append(m)
        self.marker_pub.publish(marker_array)

        pa = PoseArray()
        pa.header.frame_id = 'base_link'
        pa.poses = poses
        self.posearray_pub.publish(pa)
        self.get_logger().info(f"Published {len(poses)} waypoints for visualization.")
    
    def send_waypoints(self, poses):
        if not poses:
            self.get_logger().warn('No poses to send!')
            return

        self.get_logger().info('Waypoints:')
        for i, p in enumerate(poses):
            self.get_logger().info(
                f"{i}: x={p.position.x:.3f}, y={p.position.y:.3f}, z={p.position.z:.3f}, q=({p.orientation.x:.3f}, {p.orientation.y:.3f}, {p.orientation.z:.3f}, {p.orientation.w:.3f})"
            )

        req = GetCartesianPath.Request()
        req.header.frame_id = 'base_link'
        req.group_name = 'manipulator'
        req.link_name = 'wrist_3_link'
        req.max_step = 0.05         
        req.jump_threshold = 0.0
        req.avoid_collisions = True
        req.waypoints = []
        for p in poses:
            pose = Pose()
            pose.position.x = p.position.x
            pose.position.y = p.position.y
            pose.position.z = p.position.z
            quat = R.from_euler('xyz', [-np.pi/2, 0, 0]).as_quat()
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]
            req.waypoints.append(pose)   

        self.ik_client.wait_for_service()

        for i, pose in enumerate(poses):
            ik_req = GetPositionIK.Request()
            ik_req.ik_request.group_name = 'manipulator'
            ik_req.ik_request.robot_state.joint_state.name = []  
            ik_req.ik_request.pose_stamped = PoseStamped()
            ik_req.ik_request.pose_stamped.header.frame_id = 'base_link'
            ik_req.ik_request.pose_stamped.pose = pose
            ik_req.ik_request.ik_link_name = 'end_mill_link'
            ik_req.ik_request.timeout.sec = 1
            future = self.ik_client.call_async(ik_req)
            rclpy.spin_until_future_complete(self, future)
            result = future.result()
            ik_result = future.result()
            if result.error_code.val == result.error_code.SUCCESS:
                self.get_logger().info(f"IK solution exists for waypoint {i}")
                js = ik_result.solution.joint_state
                valid, contacts = self.check_collision(js)
                if not valid:
                    self.get_logger().warn(f"Waypoint {i} is in collision!")
                    for contact in contacts:
                            self.get_logger().warn(f"Collision between {contact.contact_body_1} and {contact.contact_body_2} (depth: {contact.depth})")
            else:
                self.get_logger().warn(f"No IK solution for waypoint {i}")

        self.get_logger().info(f'Sending {len(req.waypoints)} waypoints to compute_cartesian_path...')
        future = self.cartesian_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()

        if result is not None and result.fraction > 0.95:
            self.get_logger().info(f'Cartesian path planned for {result.fraction * 100:.1f}% of the waypoints. Executing...')
            goal_msg = ExecuteTrajectory.Goal()
            goal_msg.trajectory = result.solution
            future = self.execute_client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, future)
            exec_result = future.result()
            self.get_logger().info(f"Execution result: {exec_result}")
        else:
            self.get_logger().error(f'Cartesian path could only be planned for {result.fraction * 100:.3f}% of the waypoints!')
            # Try planning for just two points at a time
            for i in range(len(req.waypoints) - 1):
                req_test = GetCartesianPath.Request()
                req_test.header.frame_id = 'base_link'
                req_test.group_name = 'manipulator'
                req_test.link_name = 'end_mill_link'
                req_test.max_step = 0.05
                req_test.jump_threshold = 0.0
                req_test.avoid_collisions = True
                req_test.waypoints = [req.waypoints[i], req.waypoints[i + 1]]
                self.get_logger().info(f"Trying waypoint pair {i} -> {i+1}")
                test_future = self.cartesian_client.call_async(req_test)
                rclpy.spin_until_future_complete(self, test_future)
                test_result = test_future.result()
                self.get_logger().info(f"  Fraction: {test_result.fraction:.3f}")
                if test_result.fraction < 0.99:
                    self.get_logger().error(f"Path failed between waypoints {i} and {i+1}")

def main(args=None):
    rclpy.init(args=args)
    parser = GCodeParser("/home/atharva/trial_ws/src/trialpkg/gcodefiles/minimalgcode.gcode")
    poses = parser.parse()
    node = CartesianPathClient()
    node.visualize_waypoints(poses)
    node.send_waypoints(poses)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
