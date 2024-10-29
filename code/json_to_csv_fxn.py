import json

import pandas as pd


def json_to_csv(input_file_path, output_file_path):
    """
    This function will take in a data.json file and convert it into a csv.
    The following fields will be present: transcript_name, json_position, nucleotide_seq, dwelling_time_min1, sd_min1, mean_min1, dwelling_time, df, mean, dwelling_time_plus1, sd_plus1, mean_plus1

    Inputs:
        1. input_file_path: a string of the path to read
        2. output_file_path: a string of the path to read 
    
    Output:
        Returns None 
    """

    json_object = []

    # Intiialising columns 
    transcript_list = []
    position_json_list = []
    nucleotide_seq_list = []
    dwelling_time_min1_list = [] # For position -1
    sd_min1_list = []
    mean_min1_list = []
    dwelling_time_list= [] # For position 0
    sd_list = []
    mean_list = []
    dwelling_time_plus1_list = [] # For position +1
    sd_plus1_list = []
    mean_plus1_list = []

    # Reading from dataset0.json file 
    with open(input_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                json_object = json.loads(line) # loading each line of the json i.e. {"ENST00000000233":{"244":{"AAGACCA":[[0.00299,2.06,125...."
                for k, v in json_object.items(): # Key: ENST0000000023, Value: {"244":{"AAGACCA":[[0.00299,2.06,125....
                    for k2, v2 in v.items(): # Key: "244", Value: {"AAGACCA":[[0.00299,2.06,125....
                        for k3, v3 in v2.items(): # Key: "AAGACCA", Value: [[0.00299,2.06,125....
                            for read in v3: 
                                transcript_list.append(k)
                                position_json_list.append(k2)
                                nucleotide_seq_list.append(k3)
                                dwelling_time_min1_list.append(read[0]) # For position -1
                                sd_min1_list.append(read[1])
                                mean_min1_list.append(read[2])
                                dwelling_time_list.append(read[3]) # For position 0
                                sd_list.append(read[4])
                                mean_list.append(read[5])
                                dwelling_time_plus1_list.append(read[6]) # For position +1
                                sd_plus1_list.append(read[7])
                                mean_plus1_list.append(read[8])

    # Creating the dataframe 
    data = {
        'transcript_name' : transcript_list,
        'json_position': position_json_list,
        'nucleotide_seq': nucleotide_seq_list,
        'dwelling_time_min1': dwelling_time_min1_list,
        'sd_min1': sd_min1_list, 
        'mean_min1': mean_min1_list,
        'dwelling_time': dwelling_time_list,
        'sd': sd_list, 
        'mean': mean_list,
        'dwelling_time_plus1': dwelling_time_plus1_list,
        'sd_plus1': sd_plus1_list, 
        'mean_plus1': mean_plus1_list,
    }

    # Writing into csv/json 
    df = pd.DataFrame(data)
    df.to_csv(output_file_path, index = False)
    return 
