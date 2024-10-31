from pathlib import Path

import pandas as pd
import imblearn
import os
from imblearn.over_sampling import SMOTE
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


def remove_columns(df: pd.DataFrame):
    '''
    Removes transcript_name and gene_id columns.
    Add or remove columns as required.
    '''
    df1 = df.drop(["transcript_name", "gene_id", "nucleotide_seq"], axis=1)
    return df1

def create_vectorizer(df:pd.DataFrame, n):
    '''
    to vectorize nucleotide sequences using corpus from training dataset
    df should be training dataset, n would be value for ngrams
    '''
    corpus = df['nucleotide_seq']
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(n,n)).fit(corpus)
    return vectorizer

def trigram_tokenize(df:pd.DataFrame, vectorizer):
    '''
    to apply trigram on column "nucleotide_seq" using TfidfVectorizer
    '''
    v_nucleotide_seq = vectorizer.transform(df["nucleotide_seq"])
    v = pd.DataFrame(v_nucleotide_seq.toarray()) # convert sparse matrix to array

    # creating dictionary to easily add vectorized sequence features as columns to dataframe
    new_nucleotide_data = dict() 
    for i in range(v.shape[1]):
        key = "s" + str(i)
        new_nucleotide_data[key] = v.iloc[:,i]

    df_final = df.assign(**new_nucleotide_data)
    df_final = df_final.fillna(0)
    
    return df_final #returns dataframe with vectorized nucleotide features as columns

def data_split(df:pd.DataFrame):
    '''
    splits data by gene into train and test sets according to the percentage given.
    '''
    X = df.drop(["label"], axis=1)
    y = df["label"]

    gss = GroupShuffleSplit(n_splits=2, random_state=0, test_size=0.2)
    train_i, test_i = next(gss.split(X,y,groups=X.gene_id))

    X_train = X.loc[train_i]
    y_train = y.loc[train_i]

    X_test = X.loc[test_i]
    y_test = y.loc[test_i]

    return X_train, X_test, y_train, y_test

def create_standardizer(df:pd.DataFrame):
    '''
    create standardizer based on training dataset to standardize other datasets
    df should be X_train
    '''
    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(exclude=['int64', 'float64']).columns

    standardizer = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', 'passthrough', categorical_cols)  # Leave categorical columns unchanged
        ])
    
    standardizer.fit(df)

    return standardizer

def standardize_data(df:pd.DataFrame, standardizer):
    '''
    Standardizes numerical features while leaving non-numerical features unchanged using fitted preprocesser from create_standardizer function
    '''
    x_columns = df.columns
    df_scaled = standardizer.transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns = x_columns)
    
    return df_scaled

def synthetic_oversampling(X_train, y_train):
    '''
    Uses SMOTE to oversample the minority class.
    '''
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

def pipeline_all(df):
    """
    Purpose: Setting up the pipeline? to drop unnecessary columns, followed by vectorizing, splitting, standardising and oversampling
    """
    # Train-test split and remove the "transcript_name", "gene_id", "nucleotide_seq" columns
    Xtrain, Xtest, ytrain, ytest = data_split(df)

    # Vectorization
    vectorizer = create_vectorizer(Xtrain,3)
    Xtrain_v = trigram_tokenize(Xtrain, vectorizer)

    # Remove the "transcript_name", "gene_id", "nucleotide_seq" columns
    Xtrain_v = remove_columns(Xtrain_v)

    # Scaling Xtrain_v and Xtest_v with Scaler fitted on Xtrain
    standardizer = create_standardizer(Xtrain_v)
    Xtrain_scaled = standardize_data(Xtrain_v, standardizer)

    # SMOTE on training set 
    Xtrain_resampled, ytrain_resampled = synthetic_oversampling(Xtrain_scaled, ytrain)
    
    return vectorizer, standardizer, Xtrain_resampled, Xtest, ytrain_resampled, ytest

def pipeline_nn(df, v_outpath, s_outpath):
    """
    Purpose: Setting up the pipeline to drop unnecessary columns, followed by vectorizing, splitting, standardising and oversampling
    """
    # Train-test split and remove the "transcript_name", "gene_id", "nucleotide_seq" columns
    Xtrain, Xtest, ytrain, ytest = data_split(df)

    # Vectorization
    vectorizer = create_vectorizer(Xtrain,5)
    Xtrain_v = trigram_tokenize(Xtrain, vectorizer)

    # Remove the "transcript_name", "gene_id", "nucleotide_seq" columns
    Xtrain_v = remove_columns(Xtrain_v)

    # Scaling Xtrain_v and Xtest_v with Scaler fitted on Xtrain
    standardizer = create_standardizer(Xtrain_v)
    Xtrain_scaled = standardize_data(Xtrain_v, standardizer)

    # SMOTE on training set 
    Xtrain_resampled, ytrain_resampled = synthetic_oversampling(Xtrain_scaled, ytrain)

    # Merging Xtrain and Xtest, and ytrain and ytest
    Xtest_v = trigram_tokenize(Xtest, vectorizer)
    Xtest_v = remove_columns(Xtest_v)

    X_merged = pd.concat([Xtrain_resampled, Xtest_v])
    y_merged = pd.concat([ytrain_resampled, ytest])
    
    directory_v = os.path.dirname(v_outpath)
    os.makedirs(directory_v, exist_ok=True)
    directory_s = os.path.dirname(s_outpath)
    os.makedirs(directory_s, exist_ok=True)
    dump(vectorizer, v_outpath)
    dump(standardizer, s_outpath)

    return vectorizer, standardizer, X_merged, y_merged

def process_new_data(df, vectorizer, standardizer):
    """
    Purpose: Process the unlabelled, new dataset the same way as we process the labelled dataset used for training the NN model.
             Vectorization followed by dropping of unecessary columns and then standardising.

    Input:
        1. df: dataframe of the new dataset 
        2. vectorizer: vectorizer based on dataset0 training data (an output of pipeline_nn function)
        3. standardizer: standardizer based on dataset0 training data (an output of pipeline_nn function)

    Output:
        4. X_test_scaled : dataframe after processing 
    """
    
    X_test = df

    # Process test data: vectorizing, removing unwanted columns, scaling

    # Vectorization with vectorizer fitted on training data from dataset0
    X_test_v = trigram_tokenize(X_test, vectorizer)

    # Remove the "transcript_name", "nucleotide_seq" columns
    X_test_v = X_test_v.drop(["transcript_name", "nucleotide_seq"], axis=1)

    # Scaling Xtest_v with Scaler fitted on training data
    X_test_scaled = standardize_data(X_test_v, standardizer)
    
    return X_test_scaled