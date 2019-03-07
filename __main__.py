import sys
from os import listdir
from os.path import isdir, join

from grasping_learning_data_generator.orientation import get_grasping_type_learning_data, \
    transform_grasping_type_data_point_into_mln_database
from grasping_learning_data_generator.position import generate_learning_data_from_neems


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
        csv_data_frame = generate_learning_data_from_neems(path, result_dir_path)

        for experiment_file in listdir(path):
            experiment_path = join(path, experiment_file)
            transform_neem_to_mln_databases(experiment_path, result_dir_path)
    else:
        print 'A directory we the stored information is required'
