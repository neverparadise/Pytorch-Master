import os
import argparse

# argument parser : 파이썬 프로그램을 실행할 때 
# 원하는 함수나 클래스에 사용할 argument를 가져오게 해준다.
# parser 정의
parser = argparse.ArgumentParser(description='Argparse Tutorial')

# 가져올 argument들을 정의
# add_argument()를 통해 argument의 이름, 타입, 기본 값을 정의하고 도움말을 작성할 수 있다.
parser.add_argument('-n', '--name', type=str, default="홍길동", help='Name of user')
parser.add_argument('-he', '--height', type=float, default='175', help='Height of user')
parser.add_argument('-f', '--foot_size', type=int, default='270', help='Foot size of user')
parser.add_argument('-l', '--wish_list', type=list, default=[1, 2, 3, 4], help='Wish list of user')

# add_argument 함수를 실행하면 parser 객체 내부에 해당이름을 가지는 멤버변수가 생성된다. 
# parse_arg()를 통해 프로그램 실행시 argument를 입력 받는다.
args = parser.parse_args()

# 멤버 접근 연산자를 통해 argument들을 가져올 수 있다. 
# 주로 dictionary 형태로 사용한다.

user_info = {'name': args.name,
            'height': args.height, 
            'foot_size': args.foot_size,
            'wish_list': args.wish_list}

def print_user_info(**kargs):
    for key, value in kargs.items():
        print(key, value)

print_user_info(**user_info)