import argparse

from algorithm import run_default
from algorithm import run_random
from algorithm import run_TPE


def main():
    parser = argparse.ArgumentParser(description='Entrance of the program.')
    parser.add_argument('-ds', '--dataset', help='Name of the dataset', default='moodle', metavar='DS')
    parser.add_argument('-r', '--repeat', help='Number of repeats of experiments', type=int, metavar='R', default=1)
    parser.add_argument('-alg', '--algorithm', help='Optimization Algorithm',
                        choices=['Default', 'DE', 'RandomSearch', 'GridSearch', 'SMOTE', 'SMOTUNED', 'Epsilon', 'TPE'],
                        metavar='ALG', required=True)
    args = vars(parser.parse_args())

    for r in range(args['repeat']):
        if args['algorithm'] == 'Default':
            run_default.run_default(args['dataset'])
        elif args['algorithm'] == 'RandomSearch':
            run_random.run_random(args['dataset'])
        elif args['algorithm'] == "TPE":
            run_TPE.run_TPE(args['dataset'])


if __name__ == "__main__":
    main()
