import os
from os import listdir
from os.path import join
import pandas as pd
import transformations as tf
import cram2wordnet


def generate_learning_data_from_neems(neems_path, result_dir_path):
    all_csv_data_frame = get_learning_data_from_neems(neems_path)
    object_types = all_csv_data_frame['object_type'].unique()

    for object_type in object_types:
        csv_data_frame = all_csv_data_frame.loc[all_csv_data_frame['object_type'] == object_type]
        object_type = cram2wordnet.get_word_net_object(object_type)

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
    poses = pd.read_csv(poses_path, sep=';')

    object_faces_queries = reasoning_tasks.loc[reasoning_tasks['predicate'] == 'calculate-object-faces']
    grasping_tasks = pd.merge(narrative, object_faces_queries, left_on='id', right_on='action_id')
    grasping_tasks = pd.merge(grasping_tasks, poses, left_on='id_y', right_on='reasoning_task_id')

    robot_coordinate_data = {'t_x': [], 't_y': [], 't_z': []}

    for _, row in grasping_tasks[['t_x', 't_y', 't_z', 'q_x', 'q_y', 'q_z', 'q_w']].iterrows():
        translation = [row['t_x'], row['t_y'], row['t_z']]
        quaternion = [row['q_x'], row['q_y'], row['q_z'], row['q_w']]
        robot_coordinate = _transform_object_frame_to_robot_frame_(translation, quaternion)
        new_translation = tf.translation_from_matrix(robot_coordinate)
        robot_coordinate_data['t_x'].append(new_translation[0])
        robot_coordinate_data['t_y'].append(new_translation[1])
        robot_coordinate_data['t_z'].append(new_translation[2])

    grasping_tasks = grasping_tasks[['grasp', 'object_type', 'success', 'failure', 'arm', 'result']]
    grasping_tasks['t_x'] = pd.Series(robot_coordinate_data['t_x'], index=grasping_tasks.index)
    grasping_tasks['t_y'] = pd.Series(robot_coordinate_data['t_y'], index=grasping_tasks.index)
    grasping_tasks['t_z'] = pd.Series(robot_coordinate_data['t_z'], index=grasping_tasks.index)

    if grasping_tasks['failure'].dtype != 'float64':
        grasping_tasks = grasping_tasks.drop(grasping_tasks[grasping_tasks['failure'] == 'CRAM-COMMON-FAILURES:MANIPULATION-GOAL-IN-COLLISION'].index)

    grasping_tasks = grasping_tasks.dropna(axis=0, subset=['arm'])
    grasping_tasks = grasping_tasks.dropna(axis=0, subset=['grasp'])
    grasping_tasks = grasping_tasks.dropna(axis=0, subset=['result'])

    return grasping_tasks


def _transform_object_frame_to_robot_frame_(translation, quaternion):
    translation_matrix = tf.translation_matrix(translation)
    quaternion_matrix = tf.quaternion_matrix(quaternion)
    transform_matrix = tf.concatenate_matrices(translation_matrix, quaternion_matrix)

    return tf.inverse_matrix(transform_matrix)