import os
from os import listdir
from os.path import join

import pandas as pd
from high_level_markov_logic_network.fuzzy_markov_logic_network.is_a_generator import get_is_a_ground_atom
from high_level_markov_logic_network.ground_atom import GroundAtom
import cram2wordnet.cram2wordnet as cram2wordnet
import transformations as tf

__FACING_ROBOT_FACE__ = 'facing_robot_face'
__BOTTOM_FACE__ = 'bottom_face'
__OBJ_TO_BE_GRASPED__ = 'obj_to_be_grasped'
__IS_ROTATIONALLY_SYMMETRIC__ = 'is_rotationally_symmetric'
__IS_A__ = 'is_a'
__GRASP_TYPE__ = 'grasp_type'


def generate_learning_data_from_neems(neems_path, result_dir_path):
    for experiment_file in listdir(neems_path):
        experiment_path = join(neems_path, experiment_file)
        transform_neem_to_mln_databases(experiment_path, result_dir_path)


def transform_neem_to_mln_databases(neem_path, result_path):
    all_grasping_type_learning_data = get_grasping_type_learning_data(neem_path)
    object_types = all_grasping_type_learning_data['object_type'].unique()

    for object_type in object_types:
        mln_databases = []
        grasping_type_learning_data = all_grasping_type_learning_data.loc[all_grasping_type_learning_data['object_type'] == object_type]
        for _, data_point in grasping_type_learning_data.iterrows():
            mln_databases.append(transform_grasping_type_data_point_into_mln_database(data_point))

        training_file = '\n---\n'.join(mln_databases)

        mln_file_name = '{}.train.db'.format(object_type)
        mln_file_path = join(result_path, mln_file_name)

        if os.path.isfile(mln_file_path):
            with open(mln_file_path, 'a') as f:
                training_file = '\n---\n' + training_file
                f.write(training_file)
        else:
            with open(mln_file_path, 'w') as f:
                f.write(training_file)


def get_grasping_type_learning_data(neem_path):
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

    grasping_type_learning_data = pd.DataFrame()

    for picking_up_id in picking_up_ids:
        if picking_up_id.startswith('MovingToOperate'):
            grasping_action = narrative.loc[(narrative['id'] == picking_up_id)]
            grasping_action = grasping_action[['object_type', 'success']]
            grasping_action['grasp'] = ['FRONT']
        else:
            grasping_action = narrative.loc[(narrative['parent'] == picking_up_id) & (narrative['type'] == 'AcquireGraspOfSomething')]
            grasping_action = grasping_action[['grasp', 'object_type', 'success']]

        grasping_pose_features = grasping_poses.loc[(grasping_poses['action_id'] == picking_up_id)]
        robot_to_object_translation = grasping_pose_features[['t_x', 't_y', 't_z']].iloc[0].tolist()
        robot_to_object_orientation = grasping_pose_features[['q_x', 'q_y', 'q_z', 'q_w']].iloc[0].tolist()

        object_to_robot_translation = tf.get_object_robot_transformation(robot_to_object_translation, robot_to_object_orientation)
        robot_facing_face, supporting_face = tf.calculate_object_faces(object_to_robot_translation)

        grasping_action['facing_robot_face'] = robot_facing_face
        grasping_action['bottom_face'] = supporting_face

        grasping_type_learning_data = grasping_type_learning_data.append(grasping_action[['grasp', 'object_type', 'success', 'bottom_face', 'facing_robot_face']])

    return grasping_type_learning_data


def transform_grasping_type_data_point_into_mln_database(data_point):
    word_net_concept = cram2wordnet.map_cram_object_type_to_word_net_instance(data_point['object_type'])

    grasp_type_ground_atom = GroundAtom(__GRASP_TYPE__, [data_point['grasp']], float(data_point['success']))
    object_to_be_grasped_ground_atom = GroundAtom(__OBJ_TO_BE_GRASPED__, [word_net_concept])
    is_ground_atom = get_is_a_ground_atom(word_net_concept, word_net_concept)
    facing_robot_face_ground_atom = GroundAtom(__FACING_ROBOT_FACE__, [data_point['facing_robot_face']])
    bottom_face_ground_atom = GroundAtom(__BOTTOM_FACE__, [data_point['bottom_face']])

    train_database_file_content = \
        [transform_ground_atom_to_text(grasp_type_ground_atom),
         transform_ground_atom_to_text(object_to_be_grasped_ground_atom),
         transform_ground_atom_to_text(is_ground_atom),
         transform_ground_atom_to_text(facing_robot_face_ground_atom),
         transform_ground_atom_to_text(bottom_face_ground_atom)]

    return '\n'.join(train_database_file_content)


def transform_ground_atom_to_text(ground_atom):
    return '{} {}'.format(ground_atom.truth_value, ground_atom)
