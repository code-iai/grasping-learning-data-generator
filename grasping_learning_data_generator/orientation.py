from os.path import join

import pandas as pd
from high_level_markov_logic_network.fuzzy_markov_logic_network.is_a_generator import get_is_a_ground_atom
from high_level_markov_logic_network.ground_atom import GroundAtom
import cram2wordnet

__FACING_ROBOT_FACE__ = 'facing_robot_face'
__BOTTOM_FACE__ = 'bottom_face'
__OBJ_TO_BE_GRASPED__ = 'obj_to_be_grasped'
__IS_ROTATIONALLY_SYMMETRIC__ = 'is_rotationally_symmetric'
__IS_A__ = 'is_a'
__GRASP_TYPE__ = 'grasp_type'


def get_grasping_type_learning_data(neem_path):
    narrative_path = join(neem_path, 'narrative.csv')
    narrative = pd.read_csv(narrative_path, sep=';')

    reasoning_tasks_path = join(neem_path, 'reasoning_tasks.csv')
    reasoning_tasks = pd.read_csv(reasoning_tasks_path, sep=';')

    object_faces_queries = reasoning_tasks.loc[reasoning_tasks['predicate'] == 'calculate-object-faces']
    grasping_tasks = pd.merge(narrative, object_faces_queries, left_on='id', right_on='action_id')

    grasping_type_learning_data = grasping_tasks[['grasp', 'result', 'object_type', 'success']]
    grasping_type_learning_data = grasping_type_learning_data.dropna(axis=0, subset=['result'])
    facing_robot_faces = []
    bottom_faces = []

    for value in grasping_type_learning_data['result']:
        stripped_value = value.strip()
        facing_robot_face, bottom_face = stripped_value.split()

        facing_robot_faces.append(facing_robot_face)
        bottom_faces.append(bottom_face)

    grasping_type_learning_data['facing_robot_face'] = facing_robot_faces
    grasping_type_learning_data['bottom_face'] = bottom_faces

    grasping_type_learning_data.drop(['result'], axis=1, inplace=True)
    grasping_type_learning_data = grasping_type_learning_data.dropna(axis=0, subset=['grasp'])

    return grasping_type_learning_data


def transform_grasping_type_data_point_into_mln_database(data_point):
    word_net_concept = cram2wordnet.get_word_net_object(data_point['object_type'], 'object.n.01')

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