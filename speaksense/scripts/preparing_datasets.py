import argparse
import os
from speaksense.data.preparation import DataPreparation

def main():
    parser = argparse.ArgumentParser(description='Prepare datasets for SpeakSense')
    parser.add_argument('--av-speech-path', required=True, help='Path to AVSpeech dataset')
    parser.add_argument('--va-data-path', required=True, help='Path to VA interaction dataset')
    parser.add_argument('--gaze-data-path', required=True, help='Path to gaze dataset')
    parser.add_argument('--output-path', required=True, help='Output path for processed data')
    
    args = parser.parse_args()
    
    # Create data preparation instance
    data_prep = DataPreparation(args)
    
    # Prepare datasets
    data_prep.prepare_datasets()

if __name__ == "__main__":
    main()