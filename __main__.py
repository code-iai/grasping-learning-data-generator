import sys
from os import listdir
from os.path import isdir, join

import pandas as pd

from grasping_learning_data_generator.orientation import get_grasping_type_learning_data, \
    transform_grasping_type_data_point_into_mln_database
from grasping_learning_data_generator.position import get_learning_data_from_neems

_cram_to_word_net_object_ = {'BOWL':'bowl.n.01', 'CUP': 'cup.n.01', 'SPOON' : 'spoon.n.01'}


def transform_neem_to_mln_databases(neem_path, result_path):
    grasping_type_learning_data = get_grasping_type_learning_data(neem_path)
    mln_databases = []
    for _, data_point in grasping_type_learning_data.iterrows():
        mln_databases.append(transform_grasping_type_data_point_into_mln_database(data_point))

    training_file = '\n---\n'.join(mln_databases)

    with open(join(result_path, 'train.db'), 'w+') as f:
        f.write(training_file)


if __name__ == "__main__":
    args = sys.argv[1:]
    path = args[0]
    result_dir_path = args[1]

    if isdir(path):
        csv_data_frame = get_learning_data_from_neems(path)

        for experiment_file in listdir(path):
            experiment_path = join(path, experiment_file)
            transform_neem_to_mln_databases(experiment_path, result_dir_path)

        object_type = csv_data_frame['object_type'].unique()[0]
        object_type = _cram_to_word_net_object_.get(object_type,object_type)
        
        for grasping_type in csv_data_frame['grasp'].unique():
            for arm in csv_data_frame['arm'].unique():
                for faces in csv_data_frame['result'].unique():
                    grasping_type_based_grasping_tasks = csv_data_frame.loc[
                        (csv_data_frame['grasp'] == grasping_type) &
                        (csv_data_frame['arm'] == arm) &
                        (csv_data_frame['result'] == faces)]
                    grasping_type_based_grasping_tasks[['t_x', 't_y', 't_z', 'success']].to_csv(
                        join(result_dir_path, '{},{},{},{}.csv'.format(object_type, grasping_type,faces, arm)), index=False)
    else:
        print 'A directory we the stored information is required'