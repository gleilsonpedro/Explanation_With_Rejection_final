import numpy as np
import pandas as pd

def check_targets(original_set):
    
    """
    ## Check if original binary targets are following the [-1, 1] pattern.
    """
    
    original_unique = np.unique(original_set)
    print("Original Targets: ",original_unique,"\nDesired Targets: [-1,1]")
    print("Is original the desired [-1, 1]? ", np.array_equiv(original_unique,np.array([-1,1])))
    if not np.array_equiv(original_unique,np.array([-1,1])):
        if 1 in original_unique:
            print("1 exists in dataset")
            new = np.select([original_set == original_unique[0]],[-1],original_set)
        elif -1 in original_unique:
            print("-1 exists in dataset")
            new = np.select([original_set == original_unique[1]],[1],original_set)
        else:
            print("Neither exists in dataset")
            new = np.select([original_set == original_unique[0],original_set == original_unique[1]],[-1,1],original_set)
        print("New dataset targets consists of: ",np.unique(new))
        return new
    
def detail_explanation(explanations = None, patterns = None, number_of_features=None, feature_names = None, show_explanation = True, show_frequency = True, return_frequency = True):
    
    """
    ## Details the explanations obtained from the solver.
    """
    
    if explanations is None:
        raise UserWarning("explanations is None. Must pass the generated explanations.")
    if patterns is None:
        raise UserWarning("patterns is None. Must pass the patterns.")
    if number_of_features is None:
        raise UserWarning("number_of_features is None. Must pass the number of features.")
        
    columns_names = None
    relevance_df = None
    if show_frequency:
        if feature_names is not None:
            columns_names = [feature_names]
            relevance_df  = pd.DataFrame(columns=columns_names)
        else:
            columns_names = ['x'+str(i) for i in range(number_of_features)]
            relevance_df  = pd.DataFrame(columns=columns_names)
    for i, exp in enumerate(explanations):
        pattern_row = [0] * number_of_features
        if show_explanation:
            print(f"\nExplanation for Pattern {i} {patterns[i]}")
        if feature_names is None:
            for feat, val in exp:
                if show_explanation:
                    print(f"Feature [{feat}] == {val}")
                pattern_row[feat] = 1
        else:
            feats = list(zip(*explanations[i]))[0]
            vals = list(zip(*explanations[i]))[1]          
            if show_explanation:
                for feat,val in zip(feats,vals):
                    print(f"{feature_names[feat]} == {val}")
                
            for j in range(number_of_features):
                if j in feats:
                    pattern_row[j] = 1             
                else:
                    pattern_row[j] = 0
        relevance_df.loc[len(relevance_df), :] = pattern_row
    if show_frequency:
        print(relevance_df)
        print(relevance_df.sum())
    if return_frequency:
        return relevance_df
        
        
        
def minimize_risk(classifier, data_train, t1, t2, wr, labels, gap_condition = 0):
    
    """
    ## Calculations for minimizing the empiric risk.
    """
    
    solution = {'WR':0,
                'T1':0,
                'T2':0,
                'E':0,
                'R':0,
                'EWRR':0}
    index = None
    decfun = classifier.decision_function(data_train)
    n_elements = decfun.shape[0]
    #For every Rejection Weight (wr)
    for i,wr_ in enumerate(wr):        
        #For every possible  (t1, t2) values
        for j in range(0,len(t1)):
            #Get Positive, Negative and Rejected samples' indexes
            positive_indexes = np.where(decfun > t1[j])
            negative_indexes = np.where(decfun < t2[j])
            rejected_indexes = np.where((decfun <= t1[j]) & (decfun >= t2[j]))

            #Number of Rejected
            R = rejected_indexes[0].shape[0]
            if n_elements - R == 0:
                break

            #Get Number of Misclassifications
            class_p = labels[positive_indexes] #All classified as POSITIVE
            class_n = labels[negative_indexes] #All classified as NEGATIVE
            error_p = np.where(class_p == np.unique(labels)[0])[0].shape[0] #Calculate how many True Negatives were misclassified as Positives
            error_n = np.where(class_n == np.unique(labels)[1])[0].shape[0] #Calculate how many True Positives were cmislassifier as Negatives
            
            E = (error_p + error_n) 
            E_ratio =  E/(n_elements - R) #Total misclassfied ratio
            R_ratio = R/n_elements #Total Rejected ratio
            EWRR = E_ratio + wr_ * R_ratio
            if (i == 0 and j == 0) or (EWRR) < solution['EWRR']:
                solution['WR'] = wr_
                solution['T1'] = t1[j]
                solution['T2'] = t2[j]
                solution['E'] = E_ratio
                solution['R'] = R_ratio
                solution['EWRR'] = EWRR
                print(f"error_p = {error_p} error_n = {error_n}, R = {R}, E_ratio = {E_ratio}, R_ratio = {R_ratio} wr = {wr_} t1 = {t1[j]} t2 = {t2[j]} EWRR = {EWRR}")
            if R == n_elements or (error_p + error_n)/(n_elements - R) <= gap_condition:
                break
    print('Thresholds found: ',solution)
    return solution['T1'], solution['T2']             

def limits(classifier,data):
    
    """
    ## Finds the highest value (upper_limit) and lowest value (lower_limit) decision function output from the given data.
    """
    
    #Finding the data's equivalent value returned by the SVM's decision function
    dec_fun = classifier.decision_function(data)

    #Finding the superior and inferior limit of the decision function
    lim_pos = dec_fun[np.argmax(dec_fun)]
    lim_neg = dec_fun[np.argmin(dec_fun)]
    return dec_fun, lim_pos, lim_neg

def find_indexes(classifier,data, t1,t2):
    
    """
    ## Finds the predicted classes based on the given thresholds.
    """
    
    decfun = classifier.decision_function(data)
    #Find the index for all samples classified as POSITIVE (+1 class)
    positive_indexes = np.where(decfun > t1)[0]
    
    #Find the index for all samples classified as NEGATIVE (-1 class)
    negative_indexes = np.where(decfun < t2)[0]
    
    #Find the index for all samples classified as REJECTED (0 class)
    rejected_indexes = np.where((decfun <= t1) & (decfun >= t2))[0]

    return positive_indexes,negative_indexes,rejected_indexes

def find_thresholds(classifier,data_train, labels_train, wr = None):
    
    """
    ## Calculations for finding thresholds based on the given classifier, train/test data and wr value(s).
    """
    
    #Calculating all decision function values using given data, the highest and the lowest values.
    dec_fun,lim_pos,lim_neg = limits(classifier,data_train)
    print("Superior Limit: ",lim_pos,"\nInferior Limit: ",lim_neg)

    #Rejection cost, specified by the user
    if wr is None:
        wr = [0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.48]

    #Thresholds for the Positive Class (t1) and Negative Class (t2).
    t1 = []
    t2 = []
    #If the value "0" is the center, above it is the Positive Class and below the Negative Class
    #With a Reject Option problem the center becomes a range instead, with the classes being above
    #and below it, respectively.
    for i in range (1,101):
        t1.append(0.01*i*lim_pos)
        t2.append(0.01*i*lim_neg)

    #Finding the best Thresholds using the empirical risk minimization method E + wr * R
    T1,T2 = minimize_risk(classifier,data_train,t1,t2,wr, labels_train)
    
    return T1, T2

def calculate_accuracy(classifier, t1, t2, data, labels):
    
    """
    ## Calculates accuracy based on rejection thresholds
    """
    positive_indexes,negative_indexes,rejected_indexes = find_indexes(classifier, data, t1,t2)
    n_elements = len(labels)
    #Number of Rejected
    R = len(rejected_indexes)
    #Get Number of Misclassifications
    class_p = labels[positive_indexes] #All classified as POSITIVE
    class_n = labels[negative_indexes] #All classified as NEGATIVE
    error_p = np.where(class_p == np.unique(labels)[0])[0].shape[0] #Calculate how many True Negatives were misclassified as Positives
    error_n = np.where(class_n == np.unique(labels)[1])[0].shape[0] #Calculate how many True Positives were cmislassifier as Negatives
    print(f"Error p = {error_p}, Error n = {error_n}, Rejected = {R}")
    E = (error_p + error_n) 
    E_ratio =  E/(n_elements - R) #Total misclassfied ratio
    accuracy = 1 - E_ratio
    return accuracy