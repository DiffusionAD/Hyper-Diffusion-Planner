import glob
import os
from navsim.planning.training.dataset import load_feature_target_from_pickle
from matplotlib import pyplot as plt
import uuid
import numpy as np
import random
from matplotlib.axes import Axes
import matplotlib as mpl
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
import math

POLYGON_COLOR_MAP = dict(
    LANE='#DAE3F3',
    WALKWAYS='#FBE5D6',
    INTERSECTION='#D9D9D9',
)

LINE_COLOR_MAP = dict(
    LANE_CONNECTOR='#000000', 
    LANE='#000000',
)

def draw_ego_future(ax: Axes, ego_trajectory, *args, **kwargs):
    # plot ego trajectory
    ax.plot(ego_trajectory[:, 0], ego_trajectory[:, 1], *args, **kwargs)
    
def draw_neighbor_future(ax: Axes, neighbor_agents_future, *args, **kwargs):

    data = neighbor_agents_future.copy()

    mask = np.sum(np.not_equal(neighbor_agents_future[..., :8], 0), axis=-1) == 0
    neighbor_agents_future[mask] = np.nan

    for i in range(data.shape[0]):
        if data[i,0,0] != 0.:
            ax.plot(neighbor_agents_future[i, :, 0], neighbor_agents_future[i, :, 1], *args, **kwargs)
            
def draw_polygon(ax: Axes, polygons, layer_name, *args, **kwargs):

    for polygon in polygons:
        ax.fill(polygon[:, 0], polygon[:, 1], fc=POLYGON_COLOR_MAP[layer_name], ec='none')

def draw_line(ax: Axes, lines, layer_name, *args, **kwargs):
    
    if layer_name == "LANE":
        for line in lines:
            ax.plot(line[:, 0], line[:, 1], "-", color=LINE_COLOR_MAP[layer_name], lw=0.5)
    else:
        for line in lines:
            ax.plot(line[:, 0], line[:, 1], "--", color=LINE_COLOR_MAP[layer_name], lw=0.5)    

def create_ego_raster(ax, vehicle_state):
    # Extract ego vehicle dimensions
    vehicle_parameters = get_pacifica_parameters()
    ego_width = vehicle_parameters.width
    ego_front_length = vehicle_parameters.front_length
    ego_rear_length = vehicle_parameters.rear_length

    # Extract ego vehicle state
    x_center, y_center, heading = vehicle_state[0], vehicle_state[1], math.atan2(vehicle_state[3], vehicle_state[2])
    ego_bottom_right = (x_center - ego_rear_length, y_center - ego_width/2)

    # Paint the rectangle
    rect = plt.Rectangle(ego_bottom_right, ego_front_length+ego_rear_length, ego_width, linewidth=2, color='#f9c00c', alpha=1,
                        transform=mpl.transforms.Affine2D().rotate_around(*(x_center, y_center), heading) + ax.transData)
    ax.add_patch(rect)

def create_agents_raster(ax, agents, agents_label):

    for i in range(agents_label.shape[0]):
        if agents_label[i] == True:
            x_center, y_center, heading = agents[i, 0], agents[i, 1], agents[i,2]
            agent_length, agent_width = agents[i, 3],  agents[i, 4]
            agent_bottom_right = (x_center - agent_length/2, y_center - agent_width/2)

            rect = plt.Rectangle(agent_bottom_right, agent_length, agent_width, linewidth=2, color='#2C294C', alpha=1,
                                transform=mpl.transforms.Affine2D().rotate_around(*(x_center, y_center), heading) + ax.transData)
            ax.add_patch(rect)



if __name__ == "__main__":

    builder_name = 'transfuser'
    cache_dir = "/data/navsim/navsim-exp/zyn_test_cache/*/*"
    scene_list = glob.glob(cache_dir)

    print(f'total scene num: {len(scene_list)}')

    feature__name = builder_name + '_feature.gz'
    target_name = builder_name + '_target.gz'

    random.shuffle(scene_list)

    scenes = scene_list[:20]

    for scene in scenes:

        feature_path = os.path.join(scene, feature__name)
        target_path = os.path.join(scene, target_name)


        feature_dict = load_feature_target_from_pickle(feature_path)
        target_dict = load_feature_target_from_pickle(target_path)


        _, ax = plt.subplots(1, 1, figsize=(8, 8))

        for layer_name, polygons in target_dict['polygon_vector_dict'].items():
            draw_polygon(ax, polygons, layer_name)


        for layer_name, lines in target_dict['linestring_vector_dict'].items():
            draw_line(ax, lines, layer_name)

        create_ego_raster(ax, np.array([0,0,1,0], dtype=np.float32))
        create_agents_raster(ax, target_dict['agent_states'].detach().cpu().numpy(), target_dict['agent_labels'].detach().cpu().numpy())

        draw_ego_future(ax, target_dict['ego_future_trajectory'].detach().cpu().numpy(), "-", color='r', lw=2, alpha=0.5)

        draw_neighbor_future(ax, target_dict['agents_future_trajectory'].detach().cpu().numpy(), "-", color='pink', lw=1)
        ax.set_aspect('equal')
        ax.set_xlim(-32, 32)
        ax.set_ylim(-32, 32)
        plt.tight_layout()
        plt.savefig(os.path.join('./viz_test', str(uuid.uuid4())), dpi=600)

        print(feature_dict.keys())
