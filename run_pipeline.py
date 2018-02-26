import config
from train_detector import train_model, test_model_from_tfrecords
import argparse

def main(train_mode=False):
    if train_mode:
        train_model(config.Config())
    else:
        test_model_from_tfrecords(config.Config())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        dest="train_mode",
        action="store_true",
        help='flag to train the model')
    args = parser.parse_args()
    main(**vars(args))
