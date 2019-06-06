import os
from os import listdir
from os.path import join
import pandas as pd
import transformations as tf
import cram2wordnet.cram2wordnet as cram2wordnet


def generate_learning_data_from_neems(neems_path, result_dir_path):
    all_csv_data_frame = get_learning_data_from_neems(neems_path)
    object_types = all_csv_data_frame['object_type'].unique()

    for object_type in object_types:
        csv_data_frame = all_csv_data_frame.loc[all_csv_data_frame['object_type'] == object_type]
        object_type = cram2wordnet.map_cram_object_type_to_word_net_instance(object_type)

        for grasping_type in csv_data_frame['grasp'].unique():
            for arm in csv_data_frame['arm'].unique():
                for faces in csv_data_frame['result'].unique():
                    grasping_type_based_grasping_tasks = csv_data_frame.loc[
                        (csv_data_frame['grasp'] == grasping_type) &
                        (csv_data_frame['arm'] == arm) &
                        (csv_data_frame['result'] == faces)]

                    csv_file_name = '{},{},{},{}.csv'.format(object_type, grasping_type, faces, arm)
                    csv_file_path = join(result_dir_path, csv_file_name)

                    if os.path.isfile(csv_file_path):
                        with open(csv_file_path, 'a') as csv_result_file:
                            grasping_type_based_grasping_tasks[['t_x', 't_y', 't_z', 'success']].to_csv(csv_result_file,
                                                                      header=False,
                                                                      index=False)
                    else:
                        grasping_type_based_grasping_tasks[['t_x', 't_y', 't_z', 'success']].to_csv(
                            csv_file_path,
                            index=False)


def get_learning_data_from_neems(neems_path):
    learning_data_frame = pd.DataFrame()

    #neems_path/neem_name/*.csv
    for neem_folder in listdir(neems_path):
        neem_path = join(neems_path, neem_folder)
        learning_data_frame = learning_data_frame.append(get_grasping_position_learning_data(neem_path))

    return learning_data_frame


def get_grasping_position_learning_data(neem_path):
    narrative_path = join(neem_path, 'actions.csv')
    narrative = pd.read_csv(narrative_path, sep=';')

    reasoning_tasks_path = join(neem_path, 'reasoning_tasks.csv')
    reasoning_tasks = pd.read_csv(reasoning_tasks_path, sep=';')

    poses_path = join(neem_path, 'poses.csv')
    poses_tasks = pd.read_csv(poses_path, sep=';')

    object_faces_queries = \
        reasoning_tasks.loc[reasoning_tasks['predicate'] == 'cram-manipulation-interfaces:get-action-grasps']

    grasping_tasks = pd.merge(narrative, object_faces_queries, left_on='id', right_on='action_id')
    grasping_poses = pd.merge(poses_tasks, object_faces_queries, left_on='reasoning_task_id', right_on='id')
    picking_up_ids = grasping_tasks.get('id_x').unique()

    grasping_position_learning_data = pd.DataFrame()

    for picking_up_id in picking_up_ids:
        if picking_up_id.startswith('MovingToOperate'):
            grasping_action = narrative.loc[(narrative['id'] == picking_up_id)]
            grasping_action = grasping_action[['object_type', 'success', 'failure', 'arm']]
            grasping_action['grasp'] = ['FRONT']
        else:
            grasping_action = narrative.loc[(narrative['parent'] == picking_up_id) & (narrative['type'] == 'AcquireGraspOfSomething')]
            grasping_action = grasping_action[['grasp', 'object_type', 'success', 'failure', 'arm']]

        grasping_pose_features = grasping_poses.loc[(grasping_poses['action_id'] == picking_up_id)]
        robot_to_object_translation = grasping_pose_features[['t_x', 't_y', 't_z']].iloc[0].tolist()
        robot_to_object_orientation = grasping_pose_features[['q_x', 'q_y', 'q_z', 'q_w']].iloc[0].tolist()

        object_to_robot_transformation = tf.get_object_robot_transformation(robot_to_object_translation,
                                                                         robot_to_object_orientation)

        object_to_robot_translation = tf.translation_from_matrix(object_to_robot_transformation)
        robot_facing_face, supporting_face = tf.calculate_object_faces(object_to_robot_transformation)
        grasping_action['result'] = robot_facing_face + ' ' + supporting_face
        grasping_action['t_x'] = object_to_robot_translation[0]
        grasping_action['t_y'] = object_to_robot_translation[1]
        grasping_action['t_z'] = object_to_robot_translation[2]

        grasping_position_learning_data = grasping_position_learning_data.append(
            grasping_action[['grasp', 'object_type', 'success', 'result', 'failure', 'arm', 't_x', 't_y', 't_z']])

    # narrative_path = join(neem_path, 'actions.csv')
    # narrative = pd.read_csv(narrative_path, sep=';')
    #
    # reasoning_tasks_path = join(neem_path, 'reasoning_tasks.csv')
    # reasoning_tasks = pd.read_csv(reasoning_tasks_path, sep=';')
    #
    # poses_path = join(neem_path, 'poses.csv')
    # poses = pd.read_csv(poses_path, sep=';')
    #
    # object_faces_queries = reasoning_tasks.loc[reasoning_tasks['predicate'] == 'calculate-object-faces']
    # grasping_tasks = pd.merge(narrative, object_faces_queries, left_on='id', right_on='action_id')
    # grasping_tasks = pd.merge(grasping_tasks, poses, left_on='id_y', right_on='reasoning_task_id')
    #
    # robot_coordinate_data = {'t_x': [], 't_y': [], 't_z': []}
    #
    # for _, row in grasping_tasks[['t_x', 't_y', 't_z', 'q_x', 'q_y', 'q_z', 'q_w']].iterrows():
    #     translation = [row['t_x'], row['t_y'], row['t_z']]
    #     quaternion = [row['q_x'], row['q_y'], row['q_z'], row['q_w']]
    #     robot_coordinate = tf.get_object_robot_transformation(translation, quaternion)
    #     new_translation = tf.translation_from_matrix(robot_coordinate)
    #     robot_coordinate_data['t_x'].append(new_translation[0])
    #     robot_coordinate_data['t_y'].append(new_translation[1])
    #     robot_coordinate_data['t_z'].append(new_translation[2])
    #
    # grasping_tasks = grasping_tasks[['grasp', 'object_type', 'success', 'failure', 'arm', 'result']]
    # grasping_tasks['t_x'] = pd.Series(robot_coordinate_data['t_x'], index=grasping_tasks.index)
    # grasping_tasks['t_y'] = pd.Series(robot_coordinate_data['t_y'], index=grasping_tasks.index)
    # grasping_tasks['t_z'] = pd.Series(robot_coordinate_data['t_z'], index=grasping_tasks.index)

    if grasping_position_learning_data['failure'].dtype != 'float64':
        grasping_position_learning_data = grasping_position_learning_data.drop(grasping_position_learning_data[grasping_position_learning_data['failure'] == 'CRAM-COMMON-FAILURES:MANIPULATION-GOAL-IN-COLLISION'].index)

        grasping_position_learning_data = grasping_position_learning_data.dropna(axis=0, subset=['arm'])
        grasping_position_learning_data = grasping_position_learning_data.dropna(axis=0, subset=['grasp'])

    return grasping_position_learning_data
