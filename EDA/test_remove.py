import os

if __name__ == '__main__':
    path_to_file_removing = '../data/sharding_test/remove_test/test_remove_file - 副本 (4).txt'
    os.remove(path_to_file_removing)
    print('----- removed file done ! -----')