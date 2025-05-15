import pandas as pd
from sklearn import preprocessing
from math import log, sqrt

def AgeingTimeTrans(dataset):
    '''Convert ageing time to ln(sqrt(x^2 + 1))'''
    Ageing_Time = list(dataset['Ageing Time'])
    transformed = [log(sqrt(i**2+1)) for i in Ageing_Time]
    dataset['Ageing Time'] = transformed
    dataset = dataset.rename(columns = {'Ageing Time': 'Transformed Ageing Time'})
    return dataset

def RF_DataPreprocessing(dataset):
    """
    Preprocess dataset for feature engineering step with random forest
    Return:
        dataset dictionary = {property:[feature, target property],...}
    """
    
    # Convert ageing time to ln(sqrt(x^2 + 1))
    dataset = AgeingTimeTrans(dataset)
    
    # Preprocess dataset
    mech_props = ['YTS','UTS','EL']
    drop_list = ['Alloy','YTS','UTS','EL','Al (min.)','Reference','Designation','TreatmentAfterward']
    dataset_dic = {}
    for mech_prop in mech_props:
        # Create temporary drop list without one of mechanical properties
        temp_drop_list = drop_list[:]
        temp_drop_list.remove(mech_prop)     
        # Drop useless columns and rows in the dataset
        df = dataset.drop(temp_drop_list, axis=1)
        df = df.dropna() # drop row without property value
        # Create one-hot encoded feature dataset
        X = df.drop(mech_prop, axis=1)
        X = pd.get_dummies(X)
        # Normalize the feature dataset by z-score
        num_loc = 0
        for i in df.dtypes:
            if i == 'object': break
            else: num_loc += 1
        scaler = preprocessing.StandardScaler()
        scale_features = X.columns[0:num_loc]
        X[scale_features] = scaler.fit_transform(X[scale_features])
        # Create target property dataset and store X and y in dictionary
        y = df[mech_prop]    
        dataset_dic[mech_prop]=[X, y]
        
    return dataset_dic