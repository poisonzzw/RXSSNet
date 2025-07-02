import os
import yaml

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def get_save_path(save_root, args):
    root_split = save_root[0].split('/')
    # path_feature = root_split[-4] + '/' + root_split[-3] + '/' + root_split[-2]
    path_feature = root_split[-4] + '/' + root_split[-3] + '/' + 'predictions'
    save_name = root_split[-1]

    if args['is_test'] == 'True':
        if args['Data']['data_type'] == 'valid':
            save_dictionary = args['Root']['test-val_data_path'] + path_feature
            create_file(save_dictionary)
            save_path = save_dictionary + '/' + save_name + '.label'
            return save_path

        elif args['Data']['data_type'] == 'test':
            save_dictionary = args['Root']['test-test_data_path'] + path_feature
            create_file(save_dictionary)
            save_path = save_dictionary + '/' + save_name + '.label'
            return save_path
        else:
            raise Exception('data_type flag is wrong!')

    elif args['is_test'] == 'False':
        if args['Data']['data_type'] == 'valid':
            save_dictionary = args['Root']['train-val_data_path'] + path_feature
            create_file(save_dictionary)
            save_path = save_dictionary + '/' + save_name + '.label'
            return save_path

        elif args['Data']['data_type'] == 'test':
            save_dictionary = args['Root']['train-test_data_path'] + path_feature
            create_file(save_dictionary)
            save_path = save_dictionary + '/' + save_name + '.label'
            return save_path
        else:
            raise Exception('data_type flag is wrong!')

    else:
        raise Exception('is_test flag is wrong!')


def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)

def create_file(file_name):
    if not os.path.exists(file_name):
        os.makedirs(file_name)

def write_yaml(log, log_path):
    with open(os.path.join(log_path, 'Logs.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(log, stream=f, allow_unicode=True)


def write_result_yaml(log, log_path):
    with open(os.path.join(log_path, 'Result.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(log, stream=f, allow_unicode=True)


