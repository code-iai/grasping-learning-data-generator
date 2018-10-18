import pandas as pd
import sys
from os import listdir
from os.path import isdir, join

_cram_to_word_net_object_ = {'BOWL':'bowl.n.01'}

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

    facing_robot_faces = []
    bottom_faces = []

    for value in grasping_type_learning_data['result']:
        stripped_value = value.strip()
        facing_robot_face, bottom_face = stripped_value.split()

        facing_robot_faces.append(facing_robot_face)
        bottom_faces.append(bottom_face)

    grasping_type_learning_data['facing_robot_face'] = facing_robot_faces
    grasping_type_learning_data['bottom_face'] = bottom_face

    grasping_type_learning_data.drop(['result'], axis=1, inplace=True)

    return grasping_type_learning_data


def get_grasp_atom_with_truth_value(grasp, success):
    truth_value = 1.0 if success else 0.0

    return '{} {}({})\n'.format(truth_value, __GRASP_TYPE__, grasp)


def get_object_to_be_grasped_atom(object_type):
    return '1.0 {}({})\n'.format(__OBJ_TO_BE_GRASPED__, _cram_to_word_net_object_.get(object_type, 'object.n.01'))

def get_is_a_atom(object_type):
    return '1.0 {0}({1},{1})\n'.format(__IS_A__, _cram_to_word_net_object_.get(object_type, 'object.n.01'))


def get_facing_robot_face_atom(facing_robot_face):
    return '1.0 {}({})\n'.format(__FACING_ROBOT_FACE__, facing_robot_face)


def get_bottom_face_atom(bottom_face):
    return '1.0 {}({})\n'.format(__BOTTOM_FACE__, bottom_face)


def transform_grasping_type_data_point_into_mln_database(data_point):
    mln_database = get_grasp_atom_with_truth_value(data_point['grasp'], data_point['success'])
    mln_database += get_object_to_be_grasped_atom(data_point['object_type'])
    mln_database += get_is_a_atom(data_point['object_type'])
    mln_database += get_facing_robot_face_atom(data_point['facing_robot_face'])
    mln_database += get_bottom_face_atom(data_point['bottom_face'])

    return mln_database


def transform_neem_to_mln_databases(neem_path, result_path):
    grasping_type_learning_data = get_grasping_type_learning_data(neem_path)
    mln_databases = []

    for _, data_point in grasping_type_learning_data.iterrows():
        mln_databases.append(transform_grasping_type_data_point_into_mln_database(data_point))

    training_file = '---\n'.join(mln_databases)

    with open(join(result_path, 'train.db'), 'w') as f:
        f.write(training_file)


if __name__ == "__main__":
    args = sys.argv[1:]
    path = args[0]
    result_dir_path = args[1]

    if isdir(path):
        for neem_name in listdir(path):
            neem_path = join(path, neem_name)
            if isdir(neem_path):
                transform_neem_to_mln_databases(neem_path, result_dir_path)
            else:
                print transform_neem_to_mln_databases(path, result_dir_path)
    else:
        print 'A directory we the stored information is required'