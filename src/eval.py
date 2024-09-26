import argparse
import sys
import os
import yaml

# Importing the main functions from the respective scripts
from evaluate.FlashMix.eval_salsace_pose_fast import main as eval_main

def parse_args():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Select a dataset and run the corresponding training script.")
    
    # Add dataset argument
    parser.add_argument('--dataset', required=True, choices=['robotcar', 'dcc', 'vReLoc'], 
                        help="Choose the dataset: 'robotcar', 'dcc', or 'vReLoc'.")

    # Parse arguments
    return parser.parse_args()

def load_config(dataset_name):
    # Define the path to the config file
    config_file = os.path.join("src/config", f"{dataset_name}.yaml")

    # Check if the config file exists
    if not os.path.exists(config_file):
        print(f"Error: Configuration file {config_file} not found.")
        sys.exit(1)

    # Load the YAML config file
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    return config

def main():
    # Parse command-line arguments
    args = parse_args()

    # Load the appropriate config file
    config = load_config(args.dataset)
    print(config)

    # Call the respective main function based on the dataset argument
    if args.dataset == 'robotcar' or args.dataset == 'vReLoc' or args.dataset == 'dcc':
        eval_main(config, args.dataset)
    else:
        print(f"Error: Unknown dataset '{args.dataset}'")
        sys.exit(1)

if __name__ == "__main__":
    main()
