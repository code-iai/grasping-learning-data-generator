import pandas as pd
import sys
from os import listdir
from os.path import isdir, join


def transform_neem_to_mln_databases(neem_path, result_path):
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
                transform_neem_to_mln_databases(path, result_dir_path)
    else:
        print 'A directory we the stored information is required'