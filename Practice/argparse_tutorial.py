#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import argparse

# parser 정의
parser = argparse.ArgumentParser(description='Argparse Tutorial')
# add_argument()를 통해 argument의 이름, 타입, 기본 값, 도움말을 정의할 수 있다.
parser.add_argument('-k', '--korean', type=int, default=0, help="Score of korean")
parser.add_argument('-m', '--mathematics', type=int, default=0, help="Score of mathematics")
parser.add_argument('-e', '--english', type=int, default=0, help="Score of english")

# add_argument() 함수를 호출하면 parser 인스턴ㅅ 내부에 해당 이름을 가지는 멤버변수를 생성
# parse_arg()를 통해 프로그램 실행시 parser가 실행되도록 합니다.
args = parser.parse_args()

def average(args):
    total_score = 0
    total_score += args.korean
    total_score += args.mathematics
    total_score += args.english
    print(total_score / 3)
    
average(args)

