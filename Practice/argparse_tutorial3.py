
import os
import argparse

parser = argparse.ArgumentParser(description='Argparse Tutorial2')
parser.add_argument('--name', type=str, default="홍길동", help=None)
parser.add_argument('--height', type=float, default=175.0, help=None)
parser.add_argument('--foot_size', type=int, default=270, help=None)
parser.add_argument('--wish_list', type=list, default=[1, 2, 3, 4], help=None)

args = parser.parse_args()
print(type(args))
def print_user_info(args):
    print(f"{args.name}의 키는 {args.height}, 발사이즈는 {args.foot_size} 입니다.")

print_user_info(args)

