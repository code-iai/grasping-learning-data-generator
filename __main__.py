import sys
from os.path import isdir

from grasping_learning_data_generator.orientation import generate_learning_data_from_neems as generate_orientation_data
from grasping_learning_data_generator.position import generate_learning_data_from_neems as generate_position_data

if __name__ == "__main__":
    args = sys.argv[1:]
    path = args[0]
    result_dir_path = args[1]

    if isdir(path):
        #generate_position_data(path, result_dir_path)
        generate_orientation_data(path, result_dir_path)
    else:
        print 'A directory we the stored information is required'
