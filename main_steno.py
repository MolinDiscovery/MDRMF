# main_experimenter.py

import argparse
from MDRMF.experimenter import Experimenter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help="Path to the experiment YAML config.")
    args = parser.parse_args()

    exp = Experimenter(args.config)
    exp.conduct_all_experiments()

if __name__ == '__main__':
    main()