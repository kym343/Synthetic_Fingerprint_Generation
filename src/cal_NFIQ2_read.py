# --------------------------------------------------------------------------
# Tensorflow Implementation of Synthetic Fingerprint Generation
# Licensed under The MIT License [see LICENSE for details]
# Written by Young-Mook Kang
# Email: kym343@naver.com
# --------------------------------------------------------------------------
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', dest='data_path', type=str,
                    default='result.txt',
                    help='Synthetic Fingerprint test dir path')

args = parser.parse_args()


def main(data_path):
    numImgs = 7200
    f = open(data_path, 'r')
    
    i = 0
    file_list = []
    scores = np.zeros(numImgs)
    freq = np.zeros(101)

    while True:
        line = f.readline()

        if not line:
            break

        if i % 6 == 0 :
            if (i // 6) % 200 == 0:
                print('Processing {} / {}...'.format(i, numImgs))

            file_name=line[51:-1]
            file_list.append(file_name)

        if i % 6 == 3 :
            # print(line[31:-1])
            score = int(line[31:-1])
            scores[i//6] = score
            freq[int(score)]+=1

        i+=1

    f.close()

    print("===============================================================")
    for j in range(i//6):
        print("{} : {}".format(file_list[j], scores[j]))

    print("===============================================================")
    for k in range(101):
        print("{0:03d} : {1:04d}".format(k, int(freq[k])))

    print("===============================================================")
    print("total num : {}".format(sum(freq)))

if __name__ == '__main__':
    main(args.data_path)