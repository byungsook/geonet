#-*- coding: utf-8 -*-
import argparse

def str2bool(v):
    return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--image_size', type=int, default=1024)
net_arg.add_argument('--patch_size', type=int, default=128)
net_arg.add_argument('--filters', type=int, default=64)
net_arg.add_argument('--model', type=str, default='VDSR',
                     choices=['VDSR', 'DnCNN'])
net_arg.add_argument('--repeat_num', type=int, default=20)

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--data_dir', type=str, default='data')
data_arg.add_argument('--dataset', type=str, default='faces_100k_180119')
data_arg.add_argument('--batch_size', type=int, default=8)
data_arg.add_argument('--num_worker', type=int, default=16)
data_arg.add_argument('--noise_level', type=int, default=1,
                      choices=[1,2,3])
data_arg.add_argument('--range_max', type=float, default=0.125)


# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--use_gpu', type=str2bool, default=True)
train_arg.add_argument('--start_step', type=int, default=0)
train_arg.add_argument('--max_step', type=int, default=50000) # 2000
train_arg.add_argument('--lr_update_step', type=int, default=20000)
train_arg.add_argument('--lr', type=float, default=0.005)
train_arg.add_argument('--lr_lower_boundary', type=float, default=0.00001)
train_arg.add_argument('--optimizer', type=str, default='adam')
train_arg.add_argument('--beta1', type=float, default=0.5)
train_arg.add_argument('--beta2', type=float, default=0.999)

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--log_step', type=int, default=100)
misc_arg.add_argument('--test_step', type=int, default=1000)
misc_arg.add_argument('--save_sec', type=int, default=900)
misc_arg.add_argument('--log_dir', type=str, default='log')
misc_arg.add_argument('--tag', type=str, default='test')
misc_arg.add_argument('--random_seed', type=int, default=123)
misc_arg.add_argument('--load_path', type=str, default='')

def get_config():
    config, unparsed = parser.parse_known_args()
    
    if config.use_gpu:
        data_format = 'NCHW'
    else:
        data_format = 'NHWC'
    # data_format = 'NHWC' # for debug
    setattr(config, 'data_format', data_format)
    return config, unparsed
