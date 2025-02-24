import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input path", type=str)
    parser.add_argument("--output", help="output path", type=str)
    args = parser.parse_args()
    return args
