import argparse

from without_meta import train_nnpu_sigmoid
import sys
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:21'
#num_initial_pos, learning_rate, weight_decay, num_batches, seed

# #xx.py = sys.argv[0]
# #'num_initial_pos' = sys.argv[1]
# num_initial_pos = int(sys.argv[2])
# #'learning_rate' = sys.argv[3]
# learning_rate = int(sys.argv[4])
# #'weight_decay' = sys.argv[5]
# weight_decay = int(sys.argv[6])
# #'num_batches' = sys.argv[7]
# num_batches = int(sys.argv[8])
# #'seed' = sys.argv[9]
# seed = int(sys.argv[10])
# #'label_num' = sys.argv[11]
# label_num = int(sys.argv[12])
#
# #python train_nnpu_sigmoid_main.py -num_initial_pos 1000 -learning_rate 3 -weight_decay 8 -num_batches 3 -seed 1
# train_nnpu_sigmoid(num_initial_pos, learning_rate, weight_decay, num_batches, seed, label_num)

# import argparse
# from without_meta import train_nnpu_sigmoid
#
# def main():
#     # Initialize the argument parser
#     parser = argparse.ArgumentParser(description='Run train_nnpu_sigmoid with specified parameters.')
#
#     # Define expected command-line arguments
#     parser.add_argument('-num_initial_pos', type=int, required=True, help='Initial number of positive examples')
#     parser.add_argument('-learning_rate', type=float, required=True, help='Learning rate')
#     parser.add_argument('-weight_decay', type=float, required=True, help='Weight decay')
#     parser.add_argument('-num_batches', type=int, required=True, help='Number of batches')
#     parser.add_argument('-seed', type=int, required=True, help='Seed for random number generation')
#     parser.add_argument('-label_num', type=int, required=True, help='Label number')
#
#     # Parse the command-line arguments
#     args = parser.parse_args()
#
#     # Use the parsed arguments to run your function
#     train_nnpu_sigmoid(args.num_initial_pos, args.learning_rate, args.weight_decay, args.num_batches, args.seed, args.label_num)
#
# if __name__ == '__main__':
#     main()

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train NNPU with sigmoid function')
    parser.add_argument('-num_initial_pos', type=int, default=1000, help='Number of initial positive examples')
    parser.add_argument('-learning_rate', type=int, default=3, help='Learning rate for training')
    parser.add_argument('-weight_decay', type=int, default=8, help='Weight decay rate')
    parser.add_argument('-num_batches', type=int, default=3, help='Number of batches')
    parser.add_argument('-seed', type=int, default=1, help='Random seed')
    parser.add_argument('-label_num', type=int, required=True, help='Label number for training')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train_nnpu_sigmoid(args.num_initial_pos, args.learning_rate, args.weight_decay, args.num_batches, args.seed,
                       args.label_num)
