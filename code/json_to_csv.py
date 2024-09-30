import json
import pandas as pd

input_file = "dataset0.json"
output_file = "dataset0.csv"

json_object = []

# Intiialising columns for csv 
transcript_list = []
position_json_list = []
flanking_position_list = []
nucleotide_seq_list = []
five_mers_list = []
dwelling_time_list = []
sd_list = []
mean_list = []

# Reading from dataset0.json file 
with open(input_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            json_object = json.loads(line) # loading each line of the json i.e. {"ENST00000000233":{"244":{"AAGACCA":[[0.00299,2.06,125...."
            for k, v in json_object.items(): # Key: ENST0000000023, Value: {"244":{"AAGACCA":[[0.00299,2.06,125....
                for k2, v2 in v.items(): # Key: "244", Value: {"AAGACCA":[[0.00299,2.06,125....
                    for k3, v3 in v2.items(): # Key: "AAGACCA", Value: [[0.00299,2.06,125....
                        for read in v3: 
                            for i in range(0,3):
                                if i == 0:
                                    transcript_list.append(k)
                                    position_json_list.append(k2)
                                    nucleotide_seq_list.append(k3)
                                    five_mers_list.append(k3[:5])
                                    flanking_position_list.append(int(k2)-1) # Position -1 
                                    dwelling_time_list.append(read[0])
                                    sd_list.append(read[1])
                                    mean_list.append(read[2])
                                elif i == 1:
                                    transcript_list.append(k)
                                    position_json_list.append(k2)
                                    nucleotide_seq_list.append(k3)
                                    five_mers_list.append(k3[1:6])
                                    flanking_position_list.append(int(k2)) # Position 
                                    dwelling_time_list.append(read[3])
                                    sd_list.append(read[4])
                                    mean_list.append(read[5])
                                else:
                                    transcript_list.append(k)
                                    position_json_list.append(k2)
                                    nucleotide_seq_list.append(k3)
                                    five_mers_list.append(k3[2:])
                                    flanking_position_list.append(int(k2) + 1) # Position + 1
                                    dwelling_time_list.append(read[6])
                                    sd_list.append(read[7])
                                    mean_list.append(read[8])

# Creating the dataframe 
data = {
    'transcript_name' : transcript_list,
    'json_position': position_json_list,
    'flanking_position': flanking_position_list, 
    'nucleotide_seq': nucleotide_seq_list,
    'five_mers_seq': five_mers_list,
    'dwelling_time': dwelling_time_list,
    'sd': sd_list, 
    'mean': mean_list
}

df = pd.DataFrame(data)
df.to_csv(output_file, index = False)
