from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, RegisterEventHandler, IncludeLaunchDescription, SetEnvironmentVariable, ExecuteProcess, GroupAction
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution, FindExecutable
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    pkg_share = FindPackageShare('trialpkg')
    use_sim_time = LaunchConfiguration('use_sim_time')
    world_path = PathJoinSubstitution([pkg_share, 'worlds', 'empty.world'])
    xacro_file = PathJoinSubstitution([pkg_share, 'urdf', 'ur5e_corrected_with_tool0.urdf.xacro'])

    robot_description_content = Command([
        FindExecutable(name='xacro'),
        ' ',
        xacro_file,
    ])

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': ParameterValue(robot_description_content, value_type=str)
        }],
        output='screen'
    )

    # -------- RGB camera bridge --------
    bridge_rgb = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/world/empty/model/ur5e/link/wrist_3_link/sensor/rgb_camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '--ros-args', '-r',
            '/world/empty/model/ur5e/link/wrist_3_link/sensor/rgb_camera/image:=/camera/color/image_raw'
        ],
        output='screen'
    )

    delayed_bridge_rgb = TimerAction(
        period=7.0,   # adjust as needed
        actions=[bridge_rgb]
    )

    bridge_rgb_caminfo = Node(
    package='ros_gz_bridge',
    executable='parameter_bridge',
    arguments=[
        '/world/empty/model/ur5e/link/wrist_3_link/sensor/rgb_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        '--ros-args', '-r',
        '/world/empty/model/ur5e/link/wrist_3_link/sensor/rgb_camera/camera_info:=/camera/color/camera_info'
    ],
    output='screen'
    )

    delayed_bridge_caminfo = TimerAction(
        period=7.0,  # adjust for your system
        actions=[bridge_rgb_caminfo]
    )

    # -------- Depth camera bridge --------
    bridge_depth = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/world/empty/model/ur5e/link/wrist_3_link/sensor/rgbd_camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
            '--ros-args', '-r',
            '/world/empty/model/ur5e/link/wrist_3_link/sensor/rgbd_camera/depth_image:=/camera/depth/image_raw'
        ],
        output='screen'
    )

    delayed_bridge_depth = TimerAction(
        period=9.0,
        actions=[bridge_depth]
    )

    bridge_depth_caminfo = Node(
    package='ros_gz_bridge',
    executable='parameter_bridge',
    arguments=[
        '/world/empty/model/ur5e/link/wrist_3_link/sensor/rgbd_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        '--ros-args', '-r',
        '/world/empty/model/ur5e/link/wrist_3_link/sensor/rgbd_camera/camera_info:=/camera/depth/camera_info'
    ],
    output='screen'
    )

    delayed_bridge_depth_caminfo = TimerAction(
        period=9.0,
        actions=[bridge_depth_caminfo]
    )

    bridge_ft = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/world/empty/model/ur5e/joint/end_mill_joint/sensor/ft_sensor/forcetorque'
            '@geometry_msgs/msg/Wrench[gz.msgs.Wrench',
            '--ros-args', '-r',
            '/world/empty/model/ur5e/joint/end_mill_joint/sensor/ft_sensor/forcetorque:=/ft_sensor_wrench'
        ],
        output='screen'
    )

    delayed_bridge_ft = TimerAction(period=9.0, actions=[bridge_ft])

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py'])
        ]),
        launch_arguments={
            'gz_args': ['-r ', world_path],
            'on_exit_shutdown': 'true'
        }.items()
    )
    create_node = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-name', 'ur5e', '-topic', 'robot_description'],
        output='screen'
    )

    joint_state_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen'
    )

    trajectory_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_trajectory_controller', '--controller-manager', '/controller_manager'],
        output='screen'
    )

    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare('moveitpkg'), 'launch', 'move_group.launch.py'])
        ])
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            name='use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),
        gazebo,
        robot_state_publisher_node,
        create_node,
        RegisterEventHandler(
            OnProcessExit(
                target_action=create_node,
                on_exit=[
                    TimerAction(
                        period=4.0,
                        actions=[joint_state_spawner]
                    )
                ]
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=joint_state_spawner,
                on_exit=[
                    TimerAction(
                        period=4.0,
                        actions=[trajectory_controller_spawner]
                    )
                ]
            )
        ),
        moveit_launch,
        RegisterEventHandler(
            OnProcessExit(
                target_action=joint_state_spawner,
                on_exit=[
                    delayed_bridge_rgb,
                    delayed_bridge_caminfo,
                    delayed_bridge_depth,
                    delayed_bridge_depth_caminfo,
                    delayed_bridge_ft
                ]
            )
        )
    ])
    