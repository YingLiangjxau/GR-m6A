#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project    ：article title
@Description: 5-fold cross-validation of m6A mammal.
'''
print(__doc__)

import sys, argparse


def main():

    if not os.path.exists(args.output):
        print("The output path not exist! Create a new folder...\n")
        os.makedirs(args.output)
    if not os.path.exists(args.positive) or not os.path.exists(args.negative):
        print("The input train data not exist! Error\n")
        sys.exit()
    if not os.path.exists(args.ipositive) or not os.path.exists(args.inegative):
        print("The input test data not exist! Error\n")
        sys.exit()

    funciton(args.positive, args.negative, args.ipositive, args.inegative, args.output, args.fold)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Manual to the MGF6mARice')
    parser.add_argument('-tp', '--positive', type=str, help='4mC training positive data')
    parser.add_argument('-tn', '--negative', type=str, help='non-4mA training negative data')
    parser.add_argument('-ip', '--ipositive', type=str, help='4mC independent positive data')
    parser.add_argument('-in', '--inegative', type=str, help='non-4mA independent negative data')
    parser.add_argument('-f', '--fold', type=int, help='k-fold cross-validation', default=10)
    parser.add_argument('-o', '--output', type=str, help='output folder')
    args = parser.parse_args()

    from train import *
    main()

#-tp ../data/training_positive.txt -tn ../data/training_negative.txt -ip ../data/independent_positive.txt -in ../data/independent_negative.txt -f 5 -o ../result/