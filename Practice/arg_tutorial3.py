
import yaml
import os
import argparse
    
parser = argparse.ArgumentParser(description='quiz')
parser.add_argument('--config_path', type=str, default='./configs/', help='config_path')
parser.add_argument('--save_path', type=str, default='./weights/', help='save_path')
parser.add_argument('--pre_trained', type=bool, default=False, help='pre_trained')
parser.add_argument('--model_name', type=str, default='cnn.pth', help='model_name')
args = parser.parse_args()

# 1) args를 출력하세요. 
print(args)
# 2) args들 중 config_path를 통해 yaml 파일을 with open구문을 활용해 불러오고 config 변수에 할당하세요.
#    yaml.load()를 활용합니다. 
with open(args.config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# 3) config를 출력하세요. 
print(config)
# 마지막으로 셀을 저장하고 파일을 실행해보세요. 

