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
