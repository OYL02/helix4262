def json_to_csv(input_file_path, output_file_path):
    """
    This function will take in a data.json file and convert it into a csv.
    The following fields will be present: transcript_name, json_position, nucleotide_seq, dwelling_time_min1, sd_min1, mean_min1, dwelling_time, df, mean, dwelling_time_plus1, sd_plus1, mean_plus1
    The json file should be of the format: {"ENST00000000233":{"244":{"AAGACCA":[[0.00465,2.16,127.0,0.0064,3.9,127.0,0.00797,8.75,83.7],[0.0269,4.43,106.0,0.0186,10.0,123.0,0.00863,6.2,80.0],[0.00432,3.1,108.0,0.012,8.26,125.0,0.0159,2.89,78.7],[0.00996,4.52,123.0,0.0175,8.51,128.0,0.00498,2.63,80.0],[0.00764,2.81,124.0,0.00772,4.22,126.0,0.00474,5.84,80.9],[0.00432,2.17,126.0,0.0181,4.48,126.0,0.00664,2.61,78.0],[0.012,3.06,126.0,0.0054,2.66,128.0,0.00332,1.68,77.3],[0.00797,2.36,127.0,0.00391,7.06,125.0,0.00369,2.99,81.6],[0.00232,2.17,125.0,0.00894,3.98,126.0,0.00664,1.69,77.4],[0.00498,3.96,126.0,0.0116,6.89,124.0,0.00232,3.16,82.0],[0.00398,3.33,123.0,0.00972,5.17,125.0,0.00664,8.0,83.0],[0.00783,3.1,119.0,0.00345,3.95,125.0,0.00232,1.09,78.9],[0.0196,3.2,126.0,0.00797,13.0,124.0,0.00398,2.45,77.5],[0.00266,3.03,128.0,0.0127,4.55,126.0,0.00682,1.4,79.5],[0.0156,3.37,124.0,0.00775,10.4,120.0,0.00503,2.49,77.2],[0.00299,3.21,126.0,0.00853,5.53,126.0,0.00654,2.58,79.1],[0.0176,6.14,124.0,0.00863,12.1,123.0,0.00664,2.6,79.2],[0.00824,7.14,123.0,0.004,7.35,125.0,0.00354,1.56,79.9],[0.012,7.9,125.0,0.00232,3.83,128.0,0.0093,2.13,78.5],[0.0259,9.84,123.0,0.0103,9.32,126.0,0.00465,3.24,80.4],[0.0133,4.41,126.0,0.00745,3.58,127.0,0.0136,5.41,78.6],[0.0052,5.45,121.0,0.00232,7.28,127.0,0.0073,3.91,80.4],[0.00631,2.81,125.0,0.0101,8.76,125.0,0.0107,4.39,79.2],[0.00232,1.52,131.0,0.0302,8.16,126.0,0.0149,6.43,81.6],[0.013,3.92,125.0,0.00598,11.0,128.0,0.00604,6.3,83.3],[0.00498,5.03,124.0,0.00598,13.5,121.0,0.0133,1.7,73.5],[0.0146,3.58,123.0,0.00895,5.49,128.0,0.0133,6.49,80.3],[0.00963,7.26,125.0,0.0146,10.9,126.0,0.00232,3.05,81.5],[0.00863,2.06,125.0,0.00498,4.51,128.0,0.00232,1.15,79.3],[0.00365,2.06,125.0,0.00741,3.74,128.0,0.00664,4.19,79.2],[0.00896,3.7,123.0,0.00963,7.23,125.0,0.00398,3.95,81.6],[0.00851,2.9,123.0,0.0153,7.32,127.0,0.00437,1.54,79.8],[0.0259,4.53,128.0,0.00398,20.4,113.0,0.00237,1.6,78.6],[0.00668,6.07,121.0,0.00266,5.19,127.0,0.00365,1.44,83.1],[0.00631,1.96,125.0,0.0049,5.28,128.0,0.00299,1.64,80.6],[0.00797,3.1,124.0,0.00697,2.78,130.0,0.00398,1.76,79.7],[0.00266,2.95,125.0,0.00599,4.04,124.0,0.00945,2.82,77.3],[0.00963,5.39,125.0,0.0093,6.23,129.0,0.011,5.46,81.7],[0.0232,2.57,127.0,0.00414,2.14,129.0,0.00339,2.7,82.3],[0.00671,3.49,120.0,0.00244,3.3,126.0,0.0083,10.4,84.5],[0.0103,4.38,124.0,0.0073,3.98,130.0,0.00463,3.17,78.9],[0.00564,3.46,123.0,0.0119,10.4,125.0,0.0113,3.46,79.2],[0.00398,6.0,104.0,0.0173,5.78,125.0,0.00963,3.33,81.4],[0.00432,3.76,122.0,0.0147,4.64,125.0,0.0106,8.82,80.1],[0.00564,3.83,125.0,0.0398,7.25,127.0,0.00531,4.59,78.0],[0.00398,2.54,125.0,0.00498,10.7,124.0,0.0121,6.23,81.4],[0.00664,3.94,126.0,0.00868,3.72,129.0,0.0025,1.34,79.8],[0.00432,3.27,124.0,0.00822,5.3,124.0,0.00465,3.21,74.8],[0.00365,2.39,123.0,0.00555,6.03,124.0,0.00474,2.62,79.6],[0.00996,3.32,126.0,0.00398,11.1,125.0,0.00398,8.52,82.8],[0.00564,1.95,125.0,0.0138,4.63,125.0,0.00863,1.32,78.3],[0.0093,2.57,125.0,0.00556,4.74,127.0,0.0083,5.43,82.4],[0.00631,3.56,124.0,0.0133,6.11,128.0,0.0133,2.87,78.4],[0.00266,2.58,126.0,0.00365,5.88,126.0,0.00797,5.91,82.7],[0.00398,2.6,125.0,0.00417,2.22,126.0,0.00896,8.91,82.0],[0.00232,1.87,129.0,0.00842,6.24,125.0,0.00896,1.52,77.7],[0.0025,3.28,122.0,0.0209,10.5,125.0,0.00639,2.31,82.3],[0.00465,3.08,126.0,0.00642,6.22,124.0,0.0209,6.0,76.4],[0.0137,13.6,121.0,0.011,12.5,124.0,0.00631,6.13,76.6],[0.00631,5.14,108.0,0.0169,4.37,127.0,0.00863,1.72,80.1],[0.00266,1.71,124.0,0.00692,2.64,127.0,0.00531,0.933,77.8],[0.00467,7.77,123.0,0.00365,6.18,130.0,0.012,5.93,80.8],[0.0226,2.83,126.0,0.0051,11.5,117.0,0.00564,1.8,78.1],[0.00291,2.38,122.0,0.00674,4.26,124.0,0.00664,6.7,83.1],[0.0162,8.95,122.0,0.00598,6.18,128.0,0.00797,2.62,77.7],[0.0139,3.02,125.0,0.00963,13.0,124.0,0.00934,2.01,79.7],[0.0083,3.71,125.0,0.0182,14.0,121.0,0.00498,3.89,78.9],[0.00332,2.88,124.0,0.0275,5.76,126.0,0.0259,8.4,80.3],[0.00636,3.17,120.0,0.00603,5.81,127.0,0.00697,4.59,80.0],[0.00749,11.0,121.0,0.0112,11.9,122.0,0.00389,4.41,81.4],[0.00745,3.98,121.0,0.0107,7.0,124.0,0.00664,6.0,81.6],[0.0139,3.3,124.0,0.019,4.07,124.0,0.00598,9.33,82.2],[0.00496,3.84,121.0,0.0114,4.52,127.0,0.00813,4.28,79.3],[0.0249,13.1,122.0,0.00896,4.18,128.0,0.00578,2.05,78.9],[0.00821,3.23,122.0,0.00943,2.8,125.0,0.00266,3.69,81.0],[0.0103,3.42,126.0,0.0103,9.91,126.0,0.0105,1.32,79.7],[0.0093,3.1,126.0,0.0059,4.37,124.0,0.0123,1.2,76.5],[0.00697,2.71,124.0,0.00613,6.84,126.0,0.00266,1.48,73.0],[0.0133,2.63,127.0,0.00266,12.7,115.0,0.00896,2.2,79.2],[0.00266,3.03,124.0,0.014,5.75,126.0,0.00349,3.1,80.5],[0.00365,3.38,108.0,0.012,14.5,122.0,0.00474,1.45,79.8],[0.0136,4.47,106.0,0.0196,8.13,127.0,0.00498,4.06,76.1],[0.00585,2.67,124.0,0.0157,4.72,126.0,0.00398,2.26,78.0],[0.00498,5.57,125.0,0.0077,4.0,126.0,0.0093,1.75,72.6],[0.0093,3.71,126.0,0.00745,8.86,123.0,0.00398,8.32,81.8],[0.00531,4.39,126.0,0.0133,7.89,125.0,0.0123,3.91,80.4],[0.00496,3.39,123.0,0.00382,4.66,124.0,0.00797,3.0,81.0],[0.00963,3.64,126.0,0.0123,8.85,124.0,0.00299,6.79,80.7],[0.00498,3.93,124.0,0.014,4.26,126.0,0.00401,2.58,80.8],[0.00664,3.8,124.0,0.00596,6.34,126.0,0.00389,7.28,82.8],[0.00332,3.79,125.0,0.00256,3.44,127.0,0.00299,6.73,79.2],[0.00697,4.03,124.0,0.00467,8.38,121.0,0.00332,2.32,78.9],[0.00498,3.35,125.0,0.0133,6.81,127.0,0.00365,12.3,87.8],[0.00266,2.87,126.0,0.014,6.6,125.0,0.00764,5.89,82.5],[0.0113,7.56,124.0,0.00564,2.09,130.0,0.00604,4.23,81.4],[0.00598,3.11,126.0,0.00797,7.38,128.0,0.00398,2.32,80.6],[0.011,3.41,126.0,0.00481,3.95,124.0,0.00365,3.4,80.4],[0.0222,6.18,126.0,0.00764,7.84,127.0,0.00232,1.29,81.4],[0.0126,2.97,124.0,0.00682,5.22,124.0,0.00465,1.28,78.0],[0.0139,5.14,122.0,0.0155,3.62,126.0,0.00863,3.74,78.0],[0.00478,3.02,123.0,0.0232,3.66,127.0,0.00531,11.2,87.9],[0.00217,3.44,124.0,0.0122,6.09,127.0,0.0108,1.59,81.4],[0.00658,4.44,124.0,0.0216,6.14,129.0,0.014,6.62,83.8],[0.0139,3.31,126.0,0.00722,9.92,125.0,0.0083,8.85,82.4],[0.0379,2.79,127.0,0.004,1.19,130.0,0.00432,0.987,80.4],[0.00631,3.86,125.0,0.0142,5.46,127.0,0.00976,2.35,79.4],[0.0116,3.31,127.0,0.00853,8.09,125.0,0.00498,12.5,84.4],[0.00598,3.17,122.0,0.0043,3.9,126.0,0.0073,6.62,78.6],[0.0146,3.79,127.0,0.00232,6.77,121.0,0.00598,6.59,77.8],[0.0296,3.67,126.0,0.00548,3.66,125.0,0.00398,7.56,76.9],[0.0202,3.57,126.0,0.00712,7.26,126.0,0.00299,0.962,79.1],[0.00398,2.56,126.0,0.0106,11.1,124.0,0.00606,5.01,80.2],[0.00697,3.99,109.0,0.012,6.18,126.0,0.0132,7.11,80.5],[0.00432,1.76,124.0,0.00972,5.0,124.0,0.00863,1.93,78.8],[0.00564,4.37,124.0,0.00785,7.24,127.0,0.0199,5.35,78.0],[0.0083,3.59,124.0,0.00604,2.94,127.0,0.00232,3.46,83.1],[0.013,2.72,124.0,0.0103,9.6,126.0,0.00232,4.9,82.4],[0.00398,4.11,121.0,0.00976,10.6,123.0,0.00797,7.63,81.6],[0.00863,4.01,124.0,0.00816,3.51,127.0,0.00637,3.13,81.1],[0.0143,4.13,126.0,0.0208,7.59,126.0,0.00266,1.27,79.9],[0.00266,5.02,102.0,0.0093,8.89,125.0,0.00697,1.62,79.2],[0.00631,3.23,126.0,0.0128,4.33,125.0,0.00266,0.837,78.6],[0.00531,4.09,123.0,0.0093,11.8,126.0,0.00764,6.8,78.0],[0.00531,2.29,129.0,0.0123,3.97,127.0,0.00398,1.24,78.4],[0.00498,2.66,108.0,0.00863,3.34,129.0,0.00615,2.6,78.2],[0.0345,6.65,126.0,0.00332,16.3,112.0,0.00332,1.66,78.7],[0.011,3.42,124.0,0.00499,7.18,126.0,0.00812,5.06,81.9],[0.00432,3.62,109.0,0.00598,6.27,126.0,0.00566,9.39,82.6],[0.0345,10.1,125.0,0.00232,12.1,112.0,0.0176,2.05,74.6],[0.0073,6.5,124.0,0.012,8.65,127.0,0.00797,10.9,88.8],[0.00365,2.47,125.0,0.035,11.3,126.0,0.00463,6.19,79.7],[0.00299,3.25,127.0,0.0101,3.34,125.0,0.00266,1.71,79.6],[0.00337,4.71,123.0,0.016,4.34,126.0,0.00299,0.898,79.7],[0.00331,5.24,125.0,0.00531,6.34,129.0,0.00398,7.54,80.9],[0.00432,5.58,122.0,0.00504,3.06,124.0,0.00863,5.76,80.1],[0.00432,2.38,127.0,0.00634,4.82,128.0,0.00432,3.06,79.6],[0.00552,3.77,122.0,0.0121,4.98,126.0,0.00391,2.68,80.0],[0.00465,2.81,124.0,0.00988,2.46,129.0,0.0104,4.29,81.9],[0.0112,3.28,123.0,0.00403,4.3,124.0,0.00393,2.01,79.1],[0.00299,2.8,105.0,0.0153,12.8,124.0,0.00467,1.4,79.0],[0.00631,4.08,107.0,0.0179,9.57,124.0,0.00564,2.42,80.8],[0.00465,2.64,127.0,0.00332,5.04,130.0,0.00738,3.25,83.6],[0.0372,2.84,124.0,0.0533,6.56,125.0,0.0117,4.0,80.3],[0.00432,4.56,121.0,0.00877,3.18,125.0,0.00564,3.63,81.7],[0.00963,3.13,122.0,0.00332,3.35,128.0,0.00564,5.97,80.7],[0.0139,4.46,122.0,0.00564,4.81,122.0,0.00603,4.94,82.1],[0.00266,1.86,124.0,0.00963,9.29,124.0,0.00332,2.04,75.6],[0.00465,2.94,125.0,0.00863,8.11,125.0,0.00853,1.92,79.9],[0.00266,3.05,126.0,0.00478,6.64,127.0,0.00332,1.67,79.3],[0.00469,4.26,118.0,0.012,8.2,126.0,0.00232,3.08,78.3],[0.00797,4.48,126.0,0.00903,4.7,125.0,0.00631,9.28,82.8],[0.0167,8.07,124.0,0.00291,6.37,123.0,0.00365,8.2,84.2],[0.00531,2.65,130.0,0.00398,4.02,134.0,0.00299,9.13,85.2],[0.00432,3.27,124.0,0.0544,8.87,125.0,0.00291,1.35,79.8],[0.00299,2.6,129.0,0.0083,8.86,126.0,0.00232,0.731,78.0],[0.00764,5.83,123.0,0.00465,5.21,128.0,0.00299,1.89,80.3],[0.00531,4.53,117.0,0.00351,9.61,121.0,0.00337,1.22,78.4],[0.00398,2.97,119.0,0.022,6.49,123.0,0.00363,2.71,81.0],[0.00365,2.88,126.0,0.00408,6.16,126.0,0.00598,1.4,76.8],[0.00299,2.77,121.0,0.0061,3.65,124.0,0.00697,1.97,78.6],[0.0103,4.22,120.0,0.012,5.23,128.0,0.0202,3.12,82.0],[0.00432,2.82,128.0,0.00922,7.48,126.0,0.00416,2.5,79.7],[0.00432,3.49,126.0,0.0132,5.38,131.0,0.00498,3.42,81.6],[0.0123,2.86,128.0,0.00862,5.17,128.0,0.0231,4.58,78.6],[0.0116,13.7,123.0,0.0217,10.6,128.0,0.0044,1.67,80.7]]}}}
    
    Inputs:
        1. input_file_path: a string of the path to read
        2. output_file_path: a string of the path to read 
    
    Output:
        Returns None 
    """
    import json
    import pandas as pd

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

def data_agg_mean(df):
    """
    This function takes in a df that contains different read for a unique transcript name, json position and nucleotide seq and aggregate it into a single read.
    Aggregated by obtaining the mean of each group.

    Input:
        1. df: dataframe of the structure 
        transcript_name, json_position, nucleotide_seq, dwelling_time_min1, sd_min1, mean_min1, dwelling_time, df, mean, dwelling_time_plus1, sd_plus1, mean_plus1

    Output:
        1. df: dataframe of the structure 
        transcript_name, json_position, nucleotide_seq, dwelling_time_min1,sd_min1, mean_min1,dwelling_time, sd, mean,dwelling_time_plus1, sd_plus1, mean_plus1, dwelling_time_merged, sd_merged, mean_merged
    
    """

    dwelling_cols = ['dwelling_time_min1', 'dwelling_time', 'dwelling_time_plus1']
    sd_cols = ['sd_min1', 'sd', 'sd_plus1']
    mean_cols = ['mean_min1', 'mean', 'mean_plus1']

    # Group by json_position (and optionally transcript_name, nucleotide_seq), then aggregate using mean
    df_grouped = df.groupby(['json_position', 'transcript_name', 'nucleotide_seq']).agg({
        'dwelling_time_min1': 'mean',
        'dwelling_time': 'mean',
        'dwelling_time_plus1': 'mean',
        'sd_min1': 'mean',
        'sd': 'mean',
        'sd_plus1': 'mean',
        'mean_min1': 'mean',
        'mean': 'mean',
        'mean_plus1': 'mean'
    }).reset_index()

    # Merge the values by calculating the mean across each set of three columns for merged values
    df_grouped['dwelling_time_merged'] = df_grouped[dwelling_cols].mean(axis=1)
    df_grouped['sd_merged'] = df_grouped[sd_cols].mean(axis=1)
    df_grouped['mean_merged'] = df_grouped[mean_cols].mean(axis=1)

    # Keep both the individual flanking position values and the merged values
    df_final = df_grouped[['transcript_name', 'json_position', 'nucleotide_seq', 
                        'dwelling_time_min1','sd_min1', 'mean_min1',
                            'dwelling_time', 'sd', 'mean',
                            'dwelling_time_plus1', 'sd_plus1', 'mean_plus1',
                        'dwelling_time_merged', 'sd_merged', 'mean_merged']]

    return df_final
