# helix4262
project

1. start ubuntu instance (specifications to be confirmed, for now test if large works, prediction failed for t3.medium)
2. clone repo using:
https://github.com/OYL02/helix4262.git
3. cd 
4. ensure u have python3-pip installed to install python packages (about 2 minutes for installation): sudo apt install python3-pip 
5. set up environment, install python requirement with requirements.txt file using the following command:
pip install -r requirements.txt
6. create folder dataset
7. download RNA sequence into dataset folder with following command:
aws s3 cp --no-sign-request s3://sg-nex-data/data/processed_data/m6Anet/SGNex_A549_directRNA_replicate5_run1/data.json dataset
8. predict on the dataset with the following command:
python3 -m neural_net_pred -dj "../dataset/data.json"
