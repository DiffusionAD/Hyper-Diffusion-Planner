from typing import Any, Callable, List, Tuple
import io
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

from navsim.agents.abstract_agent import AbstractAgent
from navsim.evaluate.pdm_score import pdm_score
from navsim.common.dataclasses import PDMResults, Trajectory
from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
from nuplan.common.geometry.convert import absolute_to_relative_poses
from nuplan.common.actor_state.state_representation import StateSE2

from .config import BEV_PLOT_CONFIG, TRAJECTORY_CONFIG, CAMERAS_PLOT_CONFIG
from .bev import *
from .camera import *


from hdp_navsim.training.training_utils.dataclasses import Scene


def configure_bev_ax(ax: plt.Axes) -> plt.Axes:
    """
    Configure the plt ax object for birds-eye-view plots
    :param ax: matplotlib ax object
    :return: configured ax object
    """

    margin_x, margin_y = BEV_PLOT_CONFIG["figure_margin"]
    ax.set_aspect("equal")

    # NOTE: x forward, y sideways
    ax.set_xlim(-margin_y / 2, margin_y / 2)
    ax.set_ylim(-margin_x / 2, margin_x / 2)

    # NOTE: left is y positive, right is y negative
    ax.invert_xaxis()

    return ax


def configure_ax(ax: plt.Axes) -> plt.Axes:
    """
    Configure the ax object for general plotting
    :param ax: matplotlib ax object
    :return: ax object without a,y ticks
    """
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def configure_all_ax(ax: List[List[plt.Axes]]) -> List[List[plt.Axes]]:
    """
    Iterates through 2D ax list/array to apply configurations
    :param ax: 2D list/array of matplotlib ax object
    :return: configure axes
    """
    for i in range(len(ax)):
        for j in range(len(ax[i])):
            configure_ax(ax[i][j])

    return ax


def plot_bev_frame(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, plt.Axes]:
    """
    General plot for birds-eye-view visualization
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    configure_bev_ax(ax)
    configure_ax(ax)

    return fig, ax


def plot_bev_with_agent(scene: Scene, agent: AbstractAgent) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots agent and human trajectory in birds-eye-view visualization
    :param scene: navsim scene dataclass
    :param agent: navsim agent
    :return: figure and ax object of matplotlib
    """

    human_trajectory = scene.get_future_trajectory(num_trajectory_frames=8)
    agent_trajectory = agent.compute_trajectory(scene.get_agent_input())

    frame_idx = scene.scene_metadata.num_history_frames - 1
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    add_trajectory_to_bev_ax(ax, human_trajectory, TRAJECTORY_CONFIG["human"])
    add_trajectory_to_bev_ax(ax, agent_trajectory, TRAJECTORY_CONFIG["agent"])
    configure_bev_ax(ax)
    configure_ax(ax)

    return fig, ax

def plot_bev_with_agent_verbose(scene: Scene, agent: AbstractAgent, metric_cache, features) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots agent and human trajectory in birds-eye-view visualization
    :param scene: navsim scene dataclass
    :param agent: navsim agent
    :return: figure and ax object of matplotlib
    """

    human_trajectory = scene.get_future_trajectory(num_trajectory_frames=8)
    agent_trajectory = agent.compute_trajectory(features, False, use_base=False)
    base_trajectory = agent.compute_trajectory(features, False, use_base=True)

    # Use the canonical PDM rollout config -- single source of truth lives in
    # ``agent/dp_vla/utils.py`` so any change here propagates to training,
    # validation and visualisation simultaneously.
    from hdp_navsim.agent.dp_vla.utils import build_pdm_components
    simulator, scorer, proposal_sampling = build_pdm_components()

    result = pdm_score(
        metric_cache=metric_cache,
        model_trajectory=agent_trajectory,
        future_sampling=proposal_sampling,
        simulator=simulator,
        scorer=scorer,
    )
#     result = f"""
# C={result.no_at_fault_collisions}
# D={result.drivable_area_compliance}
# P={result.ego_progress}
# S={result.score}
#     """

    
    import matplotlib.image as mpimg
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    from hdp_navsim.paths import assets_dir as _assets_dir

    # Directional reference icons (L/S/R/U). Resolved from
    # ``HDP_NAVSIM_ASSETS_DIR`` so anyone reproducing the figures only
    # needs to drop the four PNGs in one directory. Missing icons are skipped
    # gracefully instead of crashing the whole plot.
    assets_dir = _assets_dir()
    logos = []
    if assets_dir is not None:
        for name in ("L.png", "S.png", "R.png", "U.png"):
            icon_path = assets_dir / "ref" / name
            if icon_path.is_file():
                logos.append(mpimg.imread(icon_path.as_posix()))
            else:
                logos.append(None)
    else:
        logos = [None, None, None, None]

    initial_ego_state = metric_cache.ego_state
    pred_trajectory = transform_trajectory(agent_trajectory, initial_ego_state)

    pred_states = get_trajectory_as_array(pred_trajectory, proposal_sampling, initial_ego_state.time_point)

    simulated_states = simulator.simulate_proposals(pred_states[None], initial_ego_state)
    simulated_states = Trajectory(simulated_states[0, 1:, :3], trajectory_sampling=proposal_sampling)

    simulated_states = absolute_to_relative_poses([StateSE2(p[0], p[1], p[2]) for p in simulated_states.poses])
    simulated_states = np.array([[s.x, s.y, s.heading] for s in simulated_states])
    simulated_states = Trajectory(simulated_states, trajectory_sampling=proposal_sampling)

    driving_command = scene.frames[0].ego_status.driving_command

    direction_idx = int(np.argmax(driving_command).item())

    frame_idx = scene.scene_metadata.num_history_frames - 1
    # fig, axes = plt.subplots(1, 4)
    fig = plt.figure(layout='constrained', figsize=(BEV_PLOT_CONFIG["figure_size"][0], BEV_PLOT_CONFIG["figure_size"][1] + CAMERAS_PLOT_CONFIG["figure_size"][1] / CAMERAS_PLOT_CONFIG["figure_size"][0] * BEV_PLOT_CONFIG["figure_size"][0] / 3))

    gs = GridSpec(2, 3, figure=fig, 
        height_ratios=[CAMERAS_PLOT_CONFIG["figure_size"][1] / CAMERAS_PLOT_CONFIG["figure_size"][0] * BEV_PLOT_CONFIG["figure_size"][0] / 3, BEV_PLOT_CONFIG["figure_size"][1]],
        wspace=0,
        hspace=0
    )
    axes = [fig.add_subplot(gs[1, :]), fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
    
    add_configured_bev_on_ax_diffusionAD(axes[0], scene.map_api, scene.frames[frame_idx])
    add_batch_trajectory_to_bev_ax_diffusionAD(axes[0], human_trajectory, TRAJECTORY_CONFIG["human"])

    pred_trajectory = transform_trajectory(base_trajectory, initial_ego_state)

    pred_states = get_trajectory_as_array(pred_trajectory, proposal_sampling, initial_ego_state.time_point)

    simulated_states = simulator.simulate_proposals(pred_states[None], initial_ego_state)
    simulated_states = Trajectory(simulated_states[0, 1:, :3], trajectory_sampling=proposal_sampling)

    simulated_states = absolute_to_relative_poses([StateSE2(p[0], p[1], p[2]) for p in simulated_states.poses])
    simulated_states = np.array([[s.x, s.y, s.heading] for s in simulated_states])
    simulated_states = Trajectory(simulated_states, trajectory_sampling=proposal_sampling)

    add_batch_trajectory_to_bev_ax_diffusionAD(axes[0], simulated_states, TRAJECTORY_CONFIG["posterior"])

    pred_trajectory = transform_trajectory(agent_trajectory, initial_ego_state)

    pred_states = get_trajectory_as_array(pred_trajectory, proposal_sampling, initial_ego_state.time_point)

    simulated_states = simulator.simulate_proposals(pred_states[None], initial_ego_state)
    simulated_states = Trajectory(simulated_states[0, 1:, :3], trajectory_sampling=proposal_sampling)

    simulated_states = absolute_to_relative_poses([StateSE2(p[0], p[1], p[2]) for p in simulated_states.poses])
    simulated_states = np.array([[s.x, s.y, s.heading] for s in simulated_states])
    simulated_states = Trajectory(simulated_states, trajectory_sampling=proposal_sampling)

    add_batch_trajectory_to_bev_ax_diffusionAD(axes[0], simulated_states, TRAJECTORY_CONFIG["agent"])
    # add_trajectory_to_bev_ax(ax, human_trajectory, TRAJECTORY_CONFIG["human"])

    axes[0].set_title("")
    # axes[0].text(-20, -25, s=direction_idx)
    axes[1].imshow(Image.open(scene.frames[0].cameras.cam_l0.image))
    axes[2].imshow(Image.open(scene.frames[0].cameras.cam_f0.image))
    axes[3].imshow(Image.open(scene.frames[0].cameras.cam_r0.image))

    # add_trajectory_to_bev_ax(ax, simulated_states, TRAJECTORY_CONFIG["posterior"])
    configure_bev_ax(axes[0])

    for ax in axes:
        configure_ax(ax)
        
    for ax in axes[1:]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    fig.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    if logos[direction_idx] is not None:
        imagebox = OffsetImage(logos[direction_idx], zoom=0.15)
        ab = AnnotationBbox(imagebox, (-20, -20), frameon=False)
        ab.set_zorder(10)
        axes[0].add_artist(ab)
    fig.canvas.draw()

    return fig, axes


def plot_bev_with_agent_batch_trajectory(scene: Scene, frame_idx: int, agent: AbstractAgent, batch_size=10) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots agent and human trajectory in birds-eye-view visualization
    :param scene: navsim scene dataclass
    :param agent: navsim agent
    :return: figure and ax object of matplotlib
    """

    agent_input = scene.get_agent_input_at_specific_frame(frame_idx)

    driving_command = agent_input.ego_statuses[3].driving_command

    directions = ['left', 'straight', 'right', 'unknown']
    direction_idx = int(np.argmax(driving_command).item())
    direction_text = directions[direction_idx]
    

    traffic_lights = scene.frames[frame_idx].traffic_lights
    traffic_lights = {lane_connector_id: is_red for lane_connector_id, is_red in traffic_lights}

    agent_trajectory = agent.compute_batch_trajectory(agent_input, batch_size)
    
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx], traffic_lights)

    ax.text(
        0.05, 0.05, 
        direction_text, 
        transform=ax.transAxes,
        fontsize=12,
        color='red',
        ha='left',  
        va='bottom' 
    )

    for trajectory in agent_trajectory:
        add_batch_trajectory_to_bev_ax(ax, trajectory, TRAJECTORY_CONFIG["agent"])
    configure_bev_ax(ax)
    configure_ax(ax)

    return fig, ax

def plot_bev_with_batch_trajectory(ax, scene: Scene, frame_idx: int, agent_trajectory) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots agent and human trajectory in birds-eye-view visualization
    :param scene: navsim scene dataclass
    :param agent: navsim agent
    :return: figure and ax object of matplotlib
    """

    agent_input = scene.get_agent_input_at_specific_frame(frame_idx)

    driving_command = agent_input.ego_statuses[3].driving_command

    directions = ['left', 'straight', 'right', 'unknown']
    direction_idx = int(np.argmax(driving_command).item())
    direction_text = directions[direction_idx]
    

    traffic_lights = scene.frames[frame_idx].traffic_lights
    traffic_lights = {lane_connector_id: is_red for lane_connector_id, is_red in traffic_lights}
    
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx], traffic_lights)

    ax.text(
        0.05, 0.05, 
        direction_text, 
        transform=ax.transAxes,
        fontsize=12,
        color='red',
        ha='left',  
        va='bottom' 
    )

    for trajectory in agent_trajectory:
        add_batch_trajectory_to_bev_ax(ax, trajectory, TRAJECTORY_CONFIG["agent"])
    configure_bev_ax(ax)
    configure_ax(ax)

    return ax


def plot_cameras_frame(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
    """
    Plots 8x cameras and birds-eye-view visualization in 3x3 grid
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """

    frame = scene.frames[frame_idx]
    fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])

    add_camera_ax(ax[0, 0], frame.cameras.cam_l0)
    add_camera_ax(ax[0, 1], frame.cameras.cam_f0)
    add_camera_ax(ax[0, 2], frame.cameras.cam_r0)

    add_camera_ax(ax[1, 0], frame.cameras.cam_l1)
    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
    add_camera_ax(ax[1, 2], frame.cameras.cam_r1)

    add_camera_ax(ax[2, 0], frame.cameras.cam_l2)
    add_camera_ax(ax[2, 1], frame.cameras.cam_b0)
    add_camera_ax(ax[2, 2], frame.cameras.cam_r2)

    configure_all_ax(ax)
    configure_bev_ax(ax[1, 1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

    return fig, ax


def plot_cameras_frame_with_lidar(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
    """
    Plots 8x cameras (including the lidar pc) and birds-eye-view visualization in 3x3 grid
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """

    frame = scene.frames[frame_idx]
    fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])

    add_lidar_to_camera_ax(ax[0, 0], frame.cameras.cam_l0, frame.lidar)
    add_lidar_to_camera_ax(ax[0, 1], frame.cameras.cam_f0, frame.lidar)
    add_lidar_to_camera_ax(ax[0, 2], frame.cameras.cam_r0, frame.lidar)

    add_lidar_to_camera_ax(ax[1, 0], frame.cameras.cam_l1, frame.lidar)
    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
    add_lidar_to_camera_ax(ax[1, 2], frame.cameras.cam_r1, frame.lidar)

    add_lidar_to_camera_ax(ax[2, 0], frame.cameras.cam_l2, frame.lidar)
    add_lidar_to_camera_ax(ax[2, 1], frame.cameras.cam_b0, frame.lidar)
    add_lidar_to_camera_ax(ax[2, 2], frame.cameras.cam_r2, frame.lidar)

    configure_all_ax(ax)
    configure_bev_ax(ax[1, 1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

    return fig, ax


def plot_cameras_frame_with_annotations(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
    """
    Plots 8x cameras (including the bounding boxes) and birds-eye-view visualization in 3x3 grid
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """

    frame = scene.frames[frame_idx]
    fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])

    add_annotations_to_camera_ax(ax[0, 0], frame.cameras.cam_l0, frame.annotations)
    add_annotations_to_camera_ax(ax[0, 1], frame.cameras.cam_f0, frame.annotations)
    add_annotations_to_camera_ax(ax[0, 2], frame.cameras.cam_r0, frame.annotations)

    add_annotations_to_camera_ax(ax[1, 0], frame.cameras.cam_l1, frame.annotations)
    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
    add_annotations_to_camera_ax(ax[1, 2], frame.cameras.cam_r1, frame.annotations)

    add_annotations_to_camera_ax(ax[2, 0], frame.cameras.cam_l2, frame.annotations)
    add_annotations_to_camera_ax(ax[2, 1], frame.cameras.cam_b0, frame.annotations)
    add_annotations_to_camera_ax(ax[2, 2], frame.cameras.cam_r2, frame.annotations)

    configure_all_ax(ax)
    configure_bev_ax(ax[1, 1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

    return fig, ax


def plot_cameras_frame_with_annotations_with_agent(scene: Scene, frame_idx: int, agent: AbstractAgent) -> Tuple[plt.Figure, Any]:
    """
    Plots 8x cameras (including the bounding boxes) and birds-eye-view visualization in 3x3 grid
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """

    assert frame_idx >= scene.scene_metadata.num_history_frames - 1 # current frame

    frame = scene.frames[frame_idx]
    fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])

    add_annotations_to_camera_ax(ax[0, 0], frame.cameras.cam_l0, frame.annotations)
    add_annotations_to_camera_ax(ax[0, 1], frame.cameras.cam_f0, frame.annotations)
    add_annotations_to_camera_ax(ax[0, 2], frame.cameras.cam_r0, frame.annotations)

    add_annotations_to_camera_ax(ax[1, 0], frame.cameras.cam_l1, frame.annotations)
    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)

    #human_trajectory = scene.get_future_trajectory()
    agent_trajectory = agent.compute_trajectory(scene.get_agent_input_at_specific_frame(frame_idx))
    # add_trajectory_to_bev_ax(ax[1, 1], human_trajectory, TRAJECTORY_CONFIG["human"])
    add_trajectory_to_bev_ax(ax[1, 1], agent_trajectory, TRAJECTORY_CONFIG["agent"])


    add_annotations_to_camera_ax(ax[1, 2], frame.cameras.cam_r1, frame.annotations)

    add_annotations_to_camera_ax(ax[2, 0], frame.cameras.cam_l2, frame.annotations)
    add_annotations_to_camera_ax(ax[2, 1], frame.cameras.cam_b0, frame.annotations)
    add_annotations_to_camera_ax(ax[2, 2], frame.cameras.cam_r2, frame.annotations)

    configure_all_ax(ax)
    configure_bev_ax(ax[1, 1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

    return fig, ax


def frame_plot_to_pil(
    callable_frame_plot: Callable[[Scene, int], Tuple[plt.Figure, Any]],
    scene: Scene,
    frame_indices: List[int],
    agent = None
) -> List[Image.Image]:
    """
    Plots a frame according to plotting function and return a list of PIL images
    :param callable_frame_plot: callable to plot a single frame
    :param scene: navsim scene dataclass
    :param frame_indices: list of indices to save
    :return: list of PIL images
    """

    images: List[Image.Image] = []

    for frame_idx in tqdm(frame_indices, desc="Rendering frames"):
        if agent is not None:
            fig, ax = callable_frame_plot(scene, frame_idx, agent)
        else:
            fig, ax = callable_frame_plot(scene, frame_idx)

        # Creating PIL image from fig
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        images.append(Image.open(buf).copy())

        # close buffer and figure
        buf.close()
        plt.close(fig)

    return images


def frame_plot_to_gif(
    file_name: str,
    callable_frame_plot: Callable[[Scene, int], Tuple[plt.Figure, Any]],
    scene: Scene,
    frame_indices: List[int],
    duration: float = 500,
    agent = None,
) -> None:
    """
    Saves a frame-wise plotting function as GIF (hard G)
    :param callable_frame_plot: callable to plot a single frame
    :param scene: navsim scene dataclass
    :param frame_indices: list of indices
    :param file_name: file path for saving to save
    :param duration: frame interval in ms, defaults to 500
    """
    images = frame_plot_to_pil(callable_frame_plot, scene, frame_indices, agent)
    images[0].save(file_name, save_all=True, append_images=images[1:], duration=duration, loop=0)
