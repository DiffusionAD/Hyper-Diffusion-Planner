from typing import Any, Dict, List
import math
import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.geometry.transform import translate_longitudinally

from navsim.common.dataclasses import Frame, Annotations, Trajectory, Lidar
from navsim.common.enums import BoundingBoxIndex, LidarIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from .lidar import filter_lidar_pc, get_lidar_pc_color
from .config import BEV_PLOT_CONFIG, MAP_LAYER_CONFIG, AGENT_CONFIG, LIDAR_CONFIG

def add_batch_trajectory_to_bev_ax_diffusionAD(ax: plt.Axes, trajectory: Trajectory, config: Dict[str, Any]) -> plt.Axes:
    poses = np.concatenate([np.array([[0, 0]]), trajectory.poses[:, :2]])
    INTERVAL = 5 if poses.shape[0] > 10 else 1
    ax.plot(
        poses[:, 1],
        poses[:, 0],
        color=config["line_color"],
        # color='#30A9DE',
        alpha=config["line_color_alpha"],  
        linewidth=2,
        linestyle=config["line_style"],
        marker='None',
        zorder=config["zorder"],
    )

    ax.plot(
        poses[-1, 1],
        poses[-1, 0],
        color=config["line_color"],
        marker='o',
        markersize=3,
        alpha=config["line_color_alpha"],
        zorder=config["zorder"] + 1,
    )

    # ax.plot(
    #     poses[-1:, 1],  
    #     poses[-1:, 0],
    #     color=config["line_color"],
    #     alpha=config["line_color_alpha"],
    #     linewidth=0,  
    #     linestyle='None',
    #     marker=config["marker"],
    #     markersize=config["marker_size"] * 0.5,
    #     markeredgecolor=config["marker_edge_color"],
    #     zorder=config["zorder"] + 1,
    # )
    return ax


def add_batch_trajectory_to_bev_ax(ax: plt.Axes, trajectory: Trajectory, config: Dict[str, Any]) -> plt.Axes:
    """
    Add trajectory poses as lint to plot
    :param ax: matplotlib ax object
    :param trajectory: navsim trajectory dataclass
    :param config: dictionary with plot parameters
    :return: ax with plot
    """
    poses = np.concatenate([np.array([[0, 0]]), trajectory.poses[:, :2]])
    ax.plot(
        poses[:, 1],
        poses[:, 0],
        color=config["line_color"],
        alpha=config["line_color_alpha"],  
        linewidth=config["line_width"] * 0.75,
        linestyle=config["line_style"],
        marker='None',
        zorder=config["zorder"],
    )


    ax.plot(
        poses[-1:, 1],  
        poses[-1:, 0],
        color=config["line_color"],
        alpha=config["line_color_alpha"],
        linewidth=0,  
        linestyle='None',
        marker=config["marker"],
        markersize=config["marker_size"] * 0.5,
        markeredgecolor=config["marker_edge_color"],
        zorder=config["zorder"] + 1,
    )
    return ax

def add_configured_bev_on_ax(ax: plt.Axes, map_api: AbstractMap, frame: Frame = None) -> plt.Axes:
    """
    Adds birds-eye-view visualization optionally with map, annotations, or lidar
    :param ax: matplotlib ax object
    :param map_api: nuPlans map interface
    :param frame: navsim frame dataclass
    :return: ax with plot
    """

    if "map" in BEV_PLOT_CONFIG["layers"]:
        add_map_to_bev_ax(ax, map_api, StateSE2(*frame.ego_status.ego_pose), frame)

    if "annotations" in BEV_PLOT_CONFIG["layers"]:
        add_annotations_to_bev_ax(ax, frame.annotations)

    if "lidar" in BEV_PLOT_CONFIG["layers"]:
        add_lidar_to_bev_ax(ax, frame.lidar)

    return ax

def add_configured_bev_on_ax_diffusionAD(ax: plt.Axes, map_api: AbstractMap, frame: Frame = None) -> plt.Axes:
    """
    Adds birds-eye-view visualization optionally with map, annotations, or lidar
    :param ax: matplotlib ax object
    :param map_api: nuPlans map interface
    :param frame: navsim frame dataclass
    :return: ax with plot
    """

    if "map" in BEV_PLOT_CONFIG["layers"]:
        add_map_to_bev_ax_diffusionAD(ax, map_api, StateSE2(*frame.ego_status.ego_pose), frame)

    if "annotations" in BEV_PLOT_CONFIG["layers"]:
        add_annotations_to_bev_ax_diffusionAD(ax, frame.annotations)

    if "lidar" in BEV_PLOT_CONFIG["layers"]:
        add_lidar_to_bev_ax(ax, frame.lidar)

    return ax


def add_annotations_to_bev_ax_diffusionAD(ax: plt.Axes, annotations: Annotations, add_ego: bool = True) -> plt.Axes:

    for name_value, box_value in zip(annotations.names, annotations.boxes):
        agent_type = tracked_object_types[name_value]
        if agent_type == TrackedObjectType.PEDESTRIAN or agent_type == TrackedObjectType.BICYCLE or agent_type == TrackedObjectType.VEHICLE:
            x, y, heading = (
                box_value[BoundingBoxIndex.X],
                box_value[BoundingBoxIndex.Y],
                box_value[BoundingBoxIndex.HEADING],
            )
            box_length, box_width, box_height = box_value[3], box_value[4], box_value[5]
            agent_box = OrientedBox(StateSE2(x, y, heading), box_length, box_width, box_height)

            if agent_type == TrackedObjectType.VEHICLE:
                add_oriented_box_to_bev_ax_agent_diffusionAD(ax, agent_box, AGENT_CONFIG[agent_type])
            else: 
                add_oriented_box_to_bev_ax_agent_diffusionAD(ax, agent_box, AGENT_CONFIG[agent_type], add_heading=False)

    if add_ego:
        car_footprint = CarFootprint.build_from_rear_axle(
            rear_axle_pose=StateSE2(0, 0, 0),
            vehicle_parameters=get_pacifica_parameters(),
        )
        add_oriented_box_to_bev_ax_ego_diffusionAD(ax, car_footprint.oriented_box, AGENT_CONFIG[TrackedObjectType.EGO])
    
    return ax

def add_annotations_to_bev_ax(ax: plt.Axes, annotations: Annotations, add_ego: bool = True) -> plt.Axes:
    """
    Adds birds-eye-view visualization of annotations (ie. bounding boxes)
    :param ax: matplotlib ax object
    :param annotations: navsim annotations dataclass
    :param add_ego: boolean weather to add ego bounding box, defaults to True
    :return: ax with plot
    """

    for name_value, box_value in zip(annotations.names, annotations.boxes):
        agent_type = tracked_object_types[name_value]

        x, y, heading = (
            box_value[BoundingBoxIndex.X],
            box_value[BoundingBoxIndex.Y],
            box_value[BoundingBoxIndex.HEADING],
        )
        box_length, box_width, box_height = box_value[3], box_value[4], box_value[5]
        agent_box = OrientedBox(StateSE2(x, y, heading), box_length, box_width, box_height)

        add_oriented_box_to_bev_ax(ax, agent_box, AGENT_CONFIG[agent_type])

    if add_ego:
        car_footprint = CarFootprint.build_from_rear_axle(
            rear_axle_pose=StateSE2(0, 0, 0),
            vehicle_parameters=get_pacifica_parameters(),
        )
        add_oriented_box_to_bev_ax(
            ax, car_footprint.oriented_box, AGENT_CONFIG[TrackedObjectType.EGO], add_heading=False
        )
    return ax

def polygon_to_patch(polygon: shapely.geometry.Polygon, origin, **kwargs):
    origin_array = origin.array
    angle = origin.heading + np.pi / 2
    rot_mat = np.array(
        [
            [np.cos(angle), np.sin(angle)],
            [np.sin(angle), -np.cos(angle)],
        ], dtype=np.float64,
    )
    polygon = np.array(polygon.exterior.xy).T
    polygon = np.matmul(polygon - origin_array, rot_mat)
    return patches.Polygon(polygon, **kwargs)


def add_map_to_bev_ax_diffusionAD(ax: plt.Axes, map_api: AbstractMap, origin: StateSE2, frame=None) -> plt.Axes:
    # layers for plotting complete layers
    polygon_layers: List[SemanticMapLayer] = [
        SemanticMapLayer.LANE,
        SemanticMapLayer.WALKWAYS,
        SemanticMapLayer.CARPARK_AREA,
        SemanticMapLayer.INTERSECTION,
        SemanticMapLayer.STOP_LINE,
        SemanticMapLayer.CROSSWALK,
    ]

    # layers for plotting complete layers
    polyline_layers: List[SemanticMapLayer] = [
        SemanticMapLayer.LANE,
        SemanticMapLayer.LANE_CONNECTOR,
    ]

    map_object_dict = map_api.get_proximal_map_objects(
        point=origin.point,
        radius=max(BEV_PLOT_CONFIG["figure_margin"]),
        layers=list(set(polygon_layers + polyline_layers)),
    )

    road_objects = (
        map_object_dict[SemanticMapLayer.LANE]
        + map_object_dict[SemanticMapLayer.LANE_CONNECTOR]
    )

    for obj in road_objects:
        obj_id = int(obj.id)
        kwargs = {"color": "#f0f5f9", "alpha": 0.6, "ec": None, "zorder": 1}
        if obj.get_roadblock_id() in frame.roadblock_ids:
            kwargs["color"] = "#c6e5d9"
            kwargs["alpha"] = 0.8
            kwargs["zorder"] = 2
        ax.add_artist(polygon_to_patch(obj.polygon, origin, **kwargs))

        if obj.get_roadblock_id() in frame.roadblock_ids:
            cl_color, linewidth = "black", 1.0
        else:
            cl_color, linewidth = "gray", 1.0
        # if traffic_light_status is not None and obj_id in tls:
            # cl_color = TRAFFIC_LIGHT_COLOR_MAPPING.get(tls[obj_id], "gray")
            # linewidth = 1
        cl = np.array([[s.x, s.y] for s in obj.baseline_path.discrete_path])

        origin_array = origin.array
        angle = origin.heading + np.pi / 2
        rot_mat = np.array(
            [
                [np.cos(angle), np.sin(angle)],
                [np.sin(angle), -np.cos(angle)],
            ], dtype=np.float64,
        )

        cl = np.matmul(cl - origin_array, rot_mat)
        ax.plot(
            cl[:, 0],
            cl[:, 1],
            color=cl_color,
            alpha=0.5,
            linestyle="--",
            zorder=3,
            linewidth=linewidth,
        )


def add_map_to_bev_ax(ax: plt.Axes, map_api: AbstractMap, origin: StateSE2, frame=None) -> plt.Axes:
    """
    Adds birds-eye-view visualization of map (ie. polygons / lines)
    TODO: add more layers for visualizations (or flags in config)
    :param ax: matplotlib ax object
    :param map_api: nuPlans map interface
    :param origin: (x,y,θ) dataclass of global ego frame
    :return: ax with plot
    """

    # layers for plotting complete layers
    polygon_layers: List[SemanticMapLayer] = [
        SemanticMapLayer.LANE,
        SemanticMapLayer.WALKWAYS,
        SemanticMapLayer.CARPARK_AREA,
        SemanticMapLayer.INTERSECTION,
        SemanticMapLayer.STOP_LINE,
        SemanticMapLayer.CROSSWALK,
    ]

    # layers for plotting complete layers
    polyline_layers: List[SemanticMapLayer] = [
        SemanticMapLayer.LANE,
        SemanticMapLayer.LANE_CONNECTOR,
    ]

    # query map api with interesting layers
    map_object_dict = map_api.get_proximal_map_objects(
        point=origin.point,
        radius=max(BEV_PLOT_CONFIG["figure_margin"]),
        layers=list(set(polygon_layers + polyline_layers)),
    )

    def _geometry_local_coords(geometry: Any, origin: StateSE2) -> Any:
        """Helper for transforming shapely geometry in coord-frame"""
        a = np.cos(origin.heading)
        b = np.sin(origin.heading)
        d = -np.sin(origin.heading)
        e = np.cos(origin.heading)
        xoff = -origin.x
        yoff = -origin.y
        translated_geometry = affinity.affine_transform(geometry, [1, 0, 0, 1, xoff, yoff])
        rotated_geometry = affinity.affine_transform(translated_geometry, [a, b, d, e, 0, 0])
        return rotated_geometry

    for polygon_layer in polygon_layers:
        for map_object in map_object_dict[polygon_layer]:
            polygon: Polygon = _geometry_local_coords(map_object.polygon, origin)
            add_polygon_to_bev_ax(ax, polygon, MAP_LAYER_CONFIG[polygon_layer])

    for polyline_layer in polyline_layers:
        for map_object in map_object_dict[polyline_layer]:
            linestring: LineString = _geometry_local_coords(map_object.baseline_path.linestring, origin)
            add_linestring_to_bev_ax(ax, linestring, MAP_LAYER_CONFIG[SemanticMapLayer.BASELINE_PATHS])
    return ax


def add_lidar_to_bev_ax(ax: plt.Axes, lidar: Lidar) -> plt.Axes:
    """
    Add lidar point cloud in birds-eye-view
    :param ax: matplotlib ax object
    :param lidar: navsim lidar dataclass
    :return: ax with plot
    """

    lidar_pc = filter_lidar_pc(lidar.lidar_pc)
    lidar_pc_colors = get_lidar_pc_color(lidar_pc, as_hex=True)
    ax.scatter(
        lidar_pc[LidarIndex.Y],
        lidar_pc[LidarIndex.X],
        c=lidar_pc_colors,
        alpha=LIDAR_CONFIG["alpha"],
        s=LIDAR_CONFIG["size"],
        zorder=LIDAR_CONFIG["zorder"],
    )
    return ax


def add_trajectory_to_bev_ax(ax: plt.Axes, trajectory: Trajectory, config: Dict[str, Any]) -> plt.Axes:
    """
    Add trajectory poses as lint to plot
    :param ax: matplotlib ax object
    :param trajectory: navsim trajectory dataclass
    :param config: dictionary with plot parameters
    :return: ax with plot
    """
    poses = np.concatenate([np.array([[0, 0]]), trajectory.poses[:, :2]])
    ax.plot(
        poses[:, 1],
        poses[:, 0],
        color=config["line_color"],
        alpha=config["line_color_alpha"],
        linewidth=config["line_width"],
        linestyle=config["line_style"],
        marker=config["marker"],
        markersize=config["marker_size"],
        markeredgecolor=config["marker_edge_color"],
        zorder=config["zorder"],
    )
    return ax


def add_oriented_box_to_bev_ax_agent_diffusionAD(
    ax: plt.Axes, box: OrientedBox, config: Dict[str, Any], add_heading: bool = True
) -> plt.Axes:
    box_corners = box.all_corners()
    corners = [[corner.x, corner.y] for corner in box_corners]
    corners = np.asarray(corners + [corners[0]])

    ax.fill(
        corners[:, 1],
        corners[:, 0],
        color=config['fill_color'],
        alpha=config["fill_color_alpha"],
        zorder=config["zorder"],
    )
    ax.plot(
        corners[:, 1],
        corners[:, 0],
        color=config['fill_color'],
        alpha=1,
        linewidth=config["line_width"],
        linestyle=config["line_style"],
        zorder=config["zorder"],
    )

    if add_heading:
        center = box.center.point
        h = box.center.heading
        x, y = center.x, center.y
        w = box.width
        
        cos_h, sin_h = math.cos(h), math.sin(h)
        
        arrow_points = np.array([
            [w/1.5, w/4, 1],
            [w/1.2, 0,   1],
            [w/1.5, -w/4, 1]
        ])
        
        transform_matrix = np.array([
            [cos_h, -sin_h, x],
            [sin_h, cos_h,  y],
            [0,     0,      1],
        ])
        
        v = np.dot(transform_matrix, arrow_points.T)
        v = v.T[:, :2]
        
        ax.plot(
            v[:, 1],
            v[:, 0],
            solid_joinstyle='miter',
            lw=1.5,
            c='#f0f5f9',
            zorder=config["zorder"] + 1
        )

    return ax

def add_oriented_box_to_bev_ax_ego_diffusionAD(
    ax: plt.Axes, box: OrientedBox, config: Dict[str, Any], add_heading: bool = True
) -> plt.Axes:
    box_corners = box.all_corners()
    corners = [[corner.x, corner.y] for corner in box_corners]
    corners = np.asarray(corners + [corners[0]])

    ax.fill(
        corners[:, 1],
        corners[:, 0],
        color='#f9c00c',
        alpha=config["fill_color_alpha"],
        zorder=config["zorder"],
    )
    ax.plot(
        corners[:, 1],
        corners[:, 0],
        color="#f9c00c",
        alpha=1,
        linewidth=config["line_width"],
        linestyle=config["line_style"],
        zorder=config["zorder"],
    )

    if add_heading:
        center = box.center.point
        h = box.center.heading
        x, y = center.x, center.y
        w = box.width
        
        cos_h, sin_h = math.cos(h), math.sin(h)
        
        arrow_points = np.array([
            [w/1.5, w/4, 1],
            [w/1.2, 0,   1],
            [w/1.5, -w/4, 1]
        ])
        
        transform_matrix = np.array([
            [cos_h, -sin_h, x],
            [sin_h, cos_h,  y],
            [0,     0,      1],
        ])
        
        v = np.dot(transform_matrix, arrow_points.T)
        v = v.T[:, :2]
        
        ax.plot(
            v[:, 1],
            v[:, 0],
            solid_joinstyle='miter',
            lw=1.5,
            c='#f0f5f9',
            zorder=config["zorder"] + 1
        )

    return ax


def add_oriented_box_to_bev_ax(
    ax: plt.Axes, box: OrientedBox, config: Dict[str, Any], add_heading: bool = True
) -> plt.Axes:
    """
    Adds birds-eye-view visualization of surrounding bounding boxes
    :param ax: matplotlib ax object
    :param box: nuPlan dataclass for 2D bounding boxes
    :param config: dictionary with plot parameters
    :param add_heading: whether to add a heading line, defaults to True
    :return: ax with plot
    """

    box_corners = box.all_corners()
    corners = [[corner.x, corner.y] for corner in box_corners]
    corners = np.asarray(corners + [corners[0]])

    ax.fill(
        corners[:, 1],
        corners[:, 0],
        color=config["fill_color"],
        alpha=config["fill_color_alpha"],
        zorder=config["zorder"],
    )
    ax.plot(
        corners[:, 1],
        corners[:, 0],
        color=config["line_color"],
        alpha=config["line_color_alpha"],
        linewidth=config["line_width"],
        linestyle=config["line_style"],
        zorder=config["zorder"],
    )

    if add_heading:
        future = translate_longitudinally(box.center, distance=box.length / 2 + 1)
        line = np.array([[box.center.x, box.center.y], [future.x, future.y]])
        ax.plot(
            line[:, 1],
            line[:, 0],
            color=config["line_color"],
            alpha=config["line_color_alpha"],
            linewidth=config["line_width"],
            linestyle=config["line_style"],
            zorder=config["zorder"],
        )

    return ax


def add_polygon_to_bev_ax(ax: plt.Axes, polygon: Polygon, config: Dict[str, Any]) -> plt.Axes:
    """
    Adds shapely polygon to birds-eye-view visualization
    :param ax: matplotlib ax object
    :param polygon: shapely Polygon
    :param config: dictionary containing plot parameters
    :return: ax with plot
    """

    def _add_element_helper(element: Polygon):
        """Helper to add single polygon to ax"""
        exterior_x, exterior_y = element.exterior.xy
        ax.fill(
            exterior_y,
            exterior_x,
            color=config["fill_color"],
            alpha=config["fill_color_alpha"],
            zorder=config["zorder"],
        )
        ax.plot(
            exterior_y,
            exterior_x,
            color=config["line_color"],
            alpha=config["line_color_alpha"],
            linewidth=config["line_width"],
            linestyle=config["line_style"],
            zorder=config["zorder"],
        )
        for interior in element.interiors:
            x_interior, y_interior = interior.xy
            ax.fill(
                y_interior,
                x_interior,
                color=BEV_PLOT_CONFIG["background_color"],
                zorder=config["zorder"],
            )
            ax.plot(
                y_interior,
                x_interior,
                color=config["line_color"],
                alpha=config["line_color_alpha"],
                linewidth=config["line_width"],
                linestyle=config["line_style"],
                zorder=config["zorder"],
            )

    if isinstance(polygon, Polygon):
        _add_element_helper(polygon)
    else:
        # NOTE: in rare cases, a map polygon has several sub-polygons.
        for element in polygon:
            _add_element_helper(element)

    return ax


def add_linestring_to_bev_ax(ax: plt.Axes, linestring: LineString, config: Dict[str, Any]) -> plt.Axes:
    """
    Adds shapely linestring (polyline) to birds-eye-view visualization
    :param ax: matplotlib ax object
    :param linestring: shapely LineString
    :param config: dictionary containing plot parameters
    :return: ax with plot
    """

    x, y = linestring.xy
    ax.plot(
        y,
        x,
        color=config["line_color"],
        alpha=config["line_color_alpha"],
        linewidth=config["line_width"],
        linestyle=config["line_style"],
        zorder=config["zorder"],
    )

    return ax
