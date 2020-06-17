import os


def get_files_dict(dir=''):
    file_dict_list = []
    for file in os.listdir(dir if dir!='' else None):
        if '.csv' in file:
            file_dict_list.append({'label': file[:-4], 'value': file})
    return file_dict_list


