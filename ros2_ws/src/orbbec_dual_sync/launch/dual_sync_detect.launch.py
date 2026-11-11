"""Launch dual Orbbec publishers + dual_view CV; vial detect OFF by default.

sync_detect (YOLO / local vial locate / vial boxes) does NOT start unless
enable_detect:=true. dual_view: furnace CV on cam0, rack CV on cam1.
Optional cam0-only furnace overlay / cam1-only rack overlay windows.
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("orbbec_dual_sync")
    default_config = os.path.join(pkg_share, "config", "dual_sync.yaml")

    config = LaunchConfiguration("config")
    preview = ParameterValue(LaunchConfiguration("preview"), value_type=bool)
    use_yolo = ParameterValue(LaunchConfiguration("use_yolo"), value_type=bool)
    enable_detect = LaunchConfiguration("enable_detect")
    enable_rack_overlay = LaunchConfiguration("enable_rack_overlay")
    enable_furnace_overlay = LaunchConfiguration("enable_furnace_overlay")
    enable_dual_view = LaunchConfiguration("enable_dual_view")
    rack_state_mode = LaunchConfiguration("rack_state_mode")
    publish_rate_hz = ParameterValue(
        LaunchConfiguration("publish_rate_hz"), value_type=float
    )
    rate_override = {"publish_rate_hz": publish_rate_hz}

    cam0 = Node(
        package="orbbec_dual_sync",
        executable="orbbec_camera_publisher",
        name="orbbec_cam0_pub",
        output="screen",
        parameters=[config, rate_override],
        respawn=True,
        respawn_delay=3.0,
    )
    cam1 = Node(
        package="orbbec_dual_sync",
        executable="orbbec_camera_publisher",
        name="orbbec_cam1_pub",
        output="screen",
        parameters=[config, rate_override],
        respawn=True,
        respawn_delay=3.0,
    )
    cam2 = Node(
        package="orbbec_dual_sync",
        executable="orbbec_camera_publisher",
        name="orbbec_cam2_pub",
        output="screen",
        # No rate_override: cam2 keeps its own lower rate + color-only from the
        # yaml so the third camera fits the remaining USB bandwidth.
        parameters=[config],
        respawn=True,
        respawn_delay=3.0,
    )
    dual_view = Node(
        package="orbbec_dual_sync",
        executable="orbbec_dual_view",
        name="orbbec_dual_view",
        output="screen",
        parameters=[
            config,
            {
                "enable_furnace_cv": True,
                "enable_rack_cv": True,
                "state_mode": rack_state_mode,
            },
        ],
        respawn=True,
        respawn_delay=3.0,
        condition=IfCondition(enable_dual_view),
    )
    furnace_overlay = Node(
        package="orbbec_dual_sync",
        executable="orbbec_furnace_overlay",
        name="orbbec_furnace_overlay",
        output="screen",
        parameters=[config],
        respawn=True,
        respawn_delay=3.0,
        condition=IfCondition(enable_furnace_overlay),
    )
    rack_overlay = Node(
        package="orbbec_dual_sync",
        executable="orbbec_rack_overlay",
        name="orbbec_rack_overlay",
        output="screen",
        parameters=[
            config,
            {"state_mode": rack_state_mode},
        ],
        respawn=True,
        respawn_delay=3.0,
        condition=IfCondition(enable_rack_overlay),
    )
    sync_detect = Node(
        package="orbbec_dual_sync",
        executable="orbbec_sync_detect",
        name="orbbec_sync_detect",
        output="screen",
        parameters=[
            config,
            {
                "show_preview": preview,
                "use_yolo": use_yolo,
                "enable_detect": ParameterValue(enable_detect, value_type=bool),
            },
        ],
        respawn=True,
        respawn_delay=3.0,
        condition=IfCondition(enable_detect),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=default_config,
                description=(
                    "YAML params installed under "
                    "share/orbbec_dual_sync/config/dual_sync.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "preview",
                default_value="true",
                description=(
                    "Show sync_detect preview when enable_detect:=true. "
                    "Ignored when detect is off."
                ),
            ),
            DeclareLaunchArgument(
                "use_yolo",
                default_value="false",
                description=(
                    "Only used when enable_detect:=true. YOLO stays opt-in."
                ),
            ),
            DeclareLaunchArgument(
                "enable_detect",
                default_value="false",
                description=(
                    "Start orbbec_sync_detect (vial YOLO/local locate). "
                    "DEFAULT false. Opt-in: enable_detect:=true"
                ),
            ),
            DeclareLaunchArgument(
                "enable_rack_overlay",
                default_value="true",
                description=(
                    "Subscribe to cam1 color and show full-rack OPEN/CLOSED "
                    "window (does not open Orbbec USB). DEFAULT true."
                ),
            ),
            DeclareLaunchArgument(
                "enable_furnace_overlay",
                default_value="false",
                description=(
                    "Optional cam0-only furnace overlay window (dual_view "
                    "already draws furnace CV on the left pane). DEFAULT false."
                ),
            ),
            DeclareLaunchArgument(
                "enable_dual_view",
                default_value="true",
                description=(
                    "Show side-by-side cam0|cam1 with furnace CV on cam0 and "
                    "rack OPEN/CLOSED CV on cam1. DEFAULT true."
                ),
            ),
            DeclareLaunchArgument(
                "rack_state_mode",
                default_value="classify",
                description=(
                    "Rack CV mode for dual_view + rack_overlay: classify "
                    "(JudgeAgent OPEN/CLOSED), force_open, or force_closed."
                ),
            ),
            DeclareLaunchArgument(
                "publish_rate_hz",
                default_value="6.0",
                description=(
                    "Timer rate for both camera publishers (default matches "
                    "dual_sync.yaml)."
                ),
            ),
            LogInfo(
                msg=(
                    "ROS dual publishers + DUAL VIEW "
                    "(cam0+furnace CV | cam1+rack CV). "
                    "Window after ~38s; furnace slots from furnace.json on "
                    "cam0; rack OPEN/CLOSED on cam1. "
                    "Disable: enable_dual_view:=false / "
                    "enable_furnace_cv:=false (yaml). "
                    "Optional cam0-only window: enable_furnace_overlay:=true. "
                    "Pin rack label: rack_state_mode:=force_closed|force_open"
                )
            ),
            cam0,
            TimerAction(period=20.0, actions=[cam1]),
            # After cam1 has the USB lock. cam2 is the third Orbbec.
            TimerAction(period=36.0, actions=[cam2]),
            # dual_view after cam2 has had time to open.
            TimerAction(period=52.0, actions=[dual_view]),
            # After cam0 publisher; furnace overlay only needs cam0 topic.
            TimerAction(period=12.0, actions=[furnace_overlay]),
            # After cam1 publisher starts; overlay only needs the topic.
            TimerAction(period=28.0, actions=[rack_overlay]),
            # Only scheduled when enable_detect:=true.
            TimerAction(period=40.0, actions=[sync_detect]),
        ]
    )
