import os
from os import listdir
from os.path import join
import pandas as pd


def generate_learning_data_from_neems(neems_path, result_dir_path):
    all_csv_data_frame = get_learning_data_from_neems(neems_path)

    for key in all_csv_data_frame.keys():
        csv_file_name = '{}.csv'.format(key)
        csv_file_path = join(result_dir_path, csv_file_name)
        all_csv_data_frame.get(key).to_csv(csv_file_path)


def get_learning_data_from_neems(neems_path):
    grasping_data_frame = pd.DataFrame()
    placing_data_frame = pd.DataFrame()
    environment_data_frame = pd.DataFrame()

    #neems_path/neem_name/*.csv
    for neem_folder in listdir(neems_path):
        neem_path = join(neems_path, neem_folder)
        new_grasping_data, new_placing_data, new_environment_data = get_position_learning_data(neem_path)
        grasping_data_frame = grasping_data_frame.append(new_grasping_data)
        placing_data_frame = placing_data_frame.append(new_placing_data)
        environment_data_frame = environment_data_frame.append(new_environment_data)

    return {'balea': grasping_data_frame, 'dish': placing_data_frame.sort_values(["t_x"])}


def get_position_learning_data(neem_path):
    narrative_path = join(neem_path, 'actions.csv')
    narrative = pd.read_csv(narrative_path, sep=';')

    poses_path = join(neem_path, 'poses.csv')
    poses_tasks = pd.read_csv(poses_path, sep=';')

    relevant_poses = pd.merge(narrative, poses_tasks, left_on='id', right_on='action_task_id')
    perceiving_actions = relevant_poses[relevant_poses['type'] != "LookingFor"]
    perceiving_actions = perceiving_actions[perceiving_actions['object_acted_on'].notna() | ((perceiving_actions['type'] == 'Grasping') & (perceiving_actions['success'] == False))]
    #perceiving_actions = perceiving_actions[perceiving_actions['success'] == True]

    environment_poses = relevant_poses[relevant_poses['information'].notna()]
    grasping_poses = perceiving_actions[perceiving_actions['object_acted_on'] == 'balea_bottle_0']
    #grasping_poses = grasping_poses[grasping_poses['bodyPartsUsed'].notna()]
    placing_poses = perceiving_actions[(perceiving_actions['object_acted_on'] == 'dish_washer_tabs_0') | ((perceiving_actions['type'] == 'Grasping') & (perceiving_actions['success'] == False)) ]
    #x_poses = list(grasping_poses.t_x)
    #y_poses = list(grasping_poses.t_y)
    # successes = set(grasping_poses.success)
    # plt.plot(x_poses, y_poses, 'ro')
    # plt.ylabel('some numbers')
    # plt.xticks(np.arange(min(x_poses), max(x_poses) + 1, 1.))
    # plt.show()

    return grasping_poses, placing_poses, environment_poses
