
import os
import argparse
import yaml

parser = argparse.ArgumentParser(description='run_cnn2')
parser.add_argument('--config_path', type=str, default='./configs/cnn.yaml', help='config file path')
args = parser.parse_args()
config_path = args.config_path

with open(config_path ,'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    print(type(config))
    
batch_size = config['batch_size']
lr = config['learning_rate']
epochs = config['epochs']
kernel_size = config['kernel_size']
stride = config['stride']

print(config)
