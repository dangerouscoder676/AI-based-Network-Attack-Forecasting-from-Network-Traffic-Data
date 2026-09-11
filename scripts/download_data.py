import os
import sys
import argparse
import subprocess
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description="Download dataset for AI Network Attack Forecasting")
    parser.add_argument('--dataset', type=str, choices=['cic-ids-2018', 'ctu-13'], required=True, help='Dataset to download')
    parser.add_argument('--output_dir', type=str, default='data/raw', help='Output directory for the raw dataset')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Preparing to download dataset: {args.dataset}")
    print(f"Target directory: {args.output_dir}\n")

    if args.dataset == 'cic-ids-2018':
        print("CIC-IDS-2018 is available via AWS S3 or Kaggle.")
        print("-" * 50)
        print("Option 1: AWS CLI (Recommended)")
        aws_command = f"aws s3 sync s3://cse-cic-ids2018/ {args.output_dir}/ --no-sign-request"
        print(f"Command: {aws_command}")
        
        print("\nOption 2: Kaggle")
        print("URL: https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv")
        print("-" * 50)
        
        if shutil.which("aws"):
            response = input("AWS CLI detected. Would you like to run the AWS download command now? (y/n): ")
            if response.lower().strip() == 'y':
                try:
                    subprocess.run(aws_command.split(), check=True)
                    print("\nDownload completed successfully.")
                except subprocess.CalledProcessError as e:
                    print(f"\nError occurred during download: {e}", file=sys.stderr)
        else:
            print("AWS CLI not found on this system. Please download manually or install the AWS CLI.")
            
    elif args.dataset == 'ctu-13':
        print("CTU-13 Dataset Scenarios:")
        print("-" * 50)
        print("The dataset consists of 13 different malware capture scenarios.")
        print("You can download the scenarios from the official source:")
        print("URL: https://www.stratosphereips.org/datasets-ctu13")
        print("\nExample links for PCAP and bidirectional flows:")
        print("- Scenario 1: https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42/")
        print("- Scenario 2: https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-43/")
        print("...")
        print("\nPlease download the required scenarios manually into the following directory:")
        print(os.path.abspath(args.output_dir))
        print("-" * 50)

    # Verify if files exist (just a basic check)
    files = os.listdir(args.output_dir)
    if files:
        print(f"\nNote: Directory {args.output_dir} currently contains {len(files)} files/folders.")

if __name__ == '__main__':
    main()
