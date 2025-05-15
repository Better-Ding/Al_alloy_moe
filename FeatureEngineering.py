
from sklearn.ensemble import RandomForestRegressor

def RF_FeatureEngineering(preprocessed_data):
    """
    Engineer low VI features with random forest
    Return:
        list: common low VI features
        dict: grouped_VIs_dict
    """
    mech_props = ['YTS','UTS','EL']
    one_hot_feature = ['ManufacturingMethod', 'HeatTreatmentMedium',
                       'StrainHardeningMethod', 'AgeingType', 'ProductionShape']
    low_VI_lists = []
    grouped_VIs_dict = {}
    for mech_prop in mech_props:
        # Load dataset of feature and target property & get feature column names
        X = preprocessed_data[mech_prop][0]
        y = preprocessed_data[mech_prop][1]
        EncodedFeature = list(X.columns)

        # Construct and train random forest model and get VIs
        clf = RandomForestRegressor(n_estimators = 1000, criterion='absolute_error', n_jobs=-1)
        clf.fit(X, y)
        gini_importance = clf.feature_importances_
        
        # Add gini importance of one-hot encoded feature together
        # ungrouped_VI = [(Feature, gini_importance), ....]
        ungrouped_VI = list(zip(EncodedFeature, gini_importance))        
        grouped_VI = []
        for feature in one_hot_feature:
            total_VI = 0
            for VI in zip(EncodedFeature, gini_importance):
                if feature in VI[0]:
                    total_VI += VI[1]
                    ungrouped_VI.remove(VI)
            grouped_VI.append((feature, total_VI))    
        grouped_VI = ungrouped_VI + grouped_VI
        # save in grouped_VIs_dict
        grouped_VIs_dict[mech_prop] = grouped_VI

        # Append grouped VI which lower than 1% into low VI list
        low_VI_list = []
        for VI in grouped_VI:
            if VI[1] < 0.01:
                low_VI_list.append(VI[0])
        
        low_VI_lists.append(low_VI_list)
    
    # Find common low VI features in all three properties
    common_low_VIs = set(low_VI_lists[0]) & set(low_VI_lists[1]) & set(low_VI_lists[2])
    print(f'Common low VI features: {common_low_VIs}')
    
    # dataset for training
    training_data = {}
    for mech_prop in mech_props:
        X = preprocessed_data[mech_prop][0]
        y = preprocessed_data[mech_prop][1]
        for low_VI in common_low_VIs:
            X = X.drop(low_VI, axis=1)
        training_data[mech_prop] = [X, y]

    
    return training_data, grouped_VIs_dict