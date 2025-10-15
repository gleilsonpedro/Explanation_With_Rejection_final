from pulp import *
from pulp import LpProblem, LpVariable, LpMinimize, LpInteger, lpSum, value, LpBinary,LpStatusOptimal
import pulp
import numpy as np
import pandas as pd
import re 
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore", message="Overwriting previously set objective.")


def rejected_validation(dual_coef, support_vectors, intercept, data, t_lower, t_upper, features_ranges, lower_bound = 0, upper_bound = 1, show_log = 0, n_threads = 1,
                                       precision = 0.0001, problem_name = "SVM_Explanation"):
    neg_validate_prob = pulp.LpProblem("Negative_Class_Validation_of_Answer_", pulp.LpMinimize)
    pos_validate_prob = pulp.LpProblem("Positive_Class_Validation_of_Answer_", pulp.LpMinimize)
    X_val_neg = np.asarray([pulp.LpVariable('xv'+str(i+1), lowBound = lower_bound, upBound = upper_bound,cat='Continuous') for i in range(len(data[0]))])
    X_val_pos = np.asarray([pulp.LpVariable('xv_'+str(i+1), lowBound = lower_bound, upBound = upper_bound,cat='Continuous') for i in range(len(data[0]))])
    neg_validate_prob += ((dual_coef @ support_vectors) @ X_val_neg.reshape(1, len(X_val_neg)).T + intercept)[0][0] <= t_lower -precision
    neg_validate_prob += 1
    pos_validate_prob += ((dual_coef @ support_vectors) @ X_val_pos.reshape(1, len(X_val_pos)).T + intercept)[0][0] >= t_upper +precision
    pos_validate_prob += 1    
    
    for i, ranges in enumerate(features_ranges):
        X_val_neg[i].lowBound = ranges[0]
        X_val_neg[i].upBound = ranges[1]
    neg_sat_val = neg_validate_prob.solve(PULP_CBC_CMD(msg=show_log, threads=n_threads,warmStart=True))
    if neg_sat_val == 1:
        print("Found intervals DO NOT maintain the class.")
        validation_values = []
        for v in X_val_neg:
            validation_values.append(v.varValue)

    for i, ranges in enumerate(features_ranges):
        X_val_pos[i].lowBound = ranges[0]
        X_val_pos[i].upBound = ranges[1]
    pos_sat_val = pos_validate_prob.solve(PULP_CBC_CMD(msg=show_log, threads=n_threads,warmStart=True))
    if pos_sat_val == 1:
        print("Found intervals DO NOT maintain the class.")
        validation_values = []
        for v in X_val_pos:
            validation_values.append(v.varValue)

    if neg_sat_val == -1 and pos_sat_val == -1:
        print("Found intervals MAINTAIN the class.")
            
def svm_explanation_rejected(dual_coef, support_vectors, intercept, data, t_lower, t_upper, lower_bound = 0, upper_bound = 1, validate = False, show_log = 0, n_threads = 1,
                            precision = 0.0001, problem_name = "SVM_Explanation"):
    """
    ## Generates explanations for rejected class.
    dual_coef - weight vector from trained SVC.
    support_vectors - support_vectors from trained SVC.
    intercept - intercept (bias) from trained SVC.
    data - patterns to be explained.
    t_lower - threshold of the negative class.
    t_upper - threshold of the positive class.
    lower_bound - lower bound of the features.
    upper_bound - upper bound of the features.
    validate - checks whether the generated explanations are valid.
    if not valid, then there is probably a problem with the given parameters.
    show_log - enables the pulp generated log.
    n_threads - enables the use of pulp multithreading.
    precision - enables bypassing the equality restriction of >= and <= by adding a small noise.
    problem_name - the name of the LP problem.
    """
    #List of generated explanations
    explanations = []
    
    #Specify the types of Optimization Problems
    neg_relevant_prob = pulp.LpProblem("Negative_Class_Features_"+problem_name, pulp.LpMinimize)
    pos_relevant_prob = pulp.LpProblem("Positive_Class_Features_"+problem_name, pulp.LpMinimize)
    

    #Specify that the features value range is between 0 and 1 (normalized dataset)
    X_neg = np.asarray([pulp.LpVariable('x'+str(i+1), lowBound = lower_bound, upBound = upper_bound,cat='Continuous') for i in range(len(data[0]))])
    X_pos = np.asarray([pulp.LpVariable('x_'+str(i+1), lowBound = lower_bound, upBound = upper_bound,cat='Continuous') for i in range(len(data[0]))])
    
    #Defines the restriction for finding if the pattern is outside the reject region
    neg_relevant_prob += ((dual_coef @ support_vectors) @ X_neg.reshape(1, len(X_neg)).T + intercept)[0][0] <= t_lower -precision #All that are inferior to the lower threshold
    pos_relevant_prob += ((dual_coef @ support_vectors) @ X_pos.reshape(1, len(X_pos)).T + intercept)[0][0] >= t_upper +precision #All that are superior to the upper threshold 
    

    #For every pattern
    for z in range(len(data)):
        relevant_features = []
        features_ranges = []
        not_relevant = []
        explanation = []
        #Setting up Prob Variables          
        for x in X_neg:
            x.lowBound = lower_bound
            x.upBound = upper_bound

        for x in X_pos:
            x.lowBound = lower_bound
            x.upBound = upper_bound

        #For every feature    
        for j in range(len(data[z])):
            #The feature to be checked
            exclude = j
            #Iterate over every feature of the pattern
            for i, feature in enumerate(data[z]):
                #If feature is relevant, keep it so that it maintains the class
                if i != exclude and i in relevant_features:
                    X_neg[i].setInitialValue(feature)
                    X_neg[i].fixValue()

                    X_pos[i].setInitialValue(feature)
                    X_pos[i].fixValue()

                #If its not the feature to be checked and haven't been worked upon yet
                elif i != exclude and i not in not_relevant and i not in relevant_features:
                    X_neg[i].setInitialValue(feature)
                    X_neg[i].fixValue()

                    X_pos[i].setInitialValue(feature)
                    X_pos[i].fixValue()

                #If feature is the one to be checked or is irrelevant    
                elif i == exclude or i in not_relevant:
                    X_neg[i].lowBound = lower_bound
                    X_neg[i].upBound = upper_bound

                    X_pos[i].lowBound = lower_bound
                    X_pos[i].upBound = upper_bound
            
            #Feature value is originally  upper/lower limited by the same value
            relevance_value = [data[z][exclude], data[z][exclude]]
            
            #Check if the feature is relevant and makes the pattern leave the reject region (negative)
            sat_neg = neg_relevant_prob.solve(PULP_CBC_CMD(msg=show_log, threads=n_threads,warmStart=True))
            if sat_neg == 1:
                values = []
                for v in X_neg:
                    values.append(v.varValue)
                relevant_features.append(exclude)
                features_ranges.append(relevance_value)
                explanation.append((exclude, data[z][exclude]))
            else:
                #Check if the feature is relevant and makes the pattern leave the reject region (positive)
                sat_pos = pos_relevant_prob.solve(PULP_CBC_CMD(msg=show_log, threads=n_threads,warmStart=True))
                if sat_pos == 1:
                    values = []
                    for v in X_pos:
                        values.append(v.varValue)
                    relevant_features.append(exclude)
                    features_ranges.append(relevance_value)
                    explanation.append((exclude, data[z][exclude]))
                else:
                    #Checked feature is not relevant and does not make the pattern leave the reject region
                    not_relevant.append(exclude)
                    features_ranges.append([0,1])
                    
        #Validate explanations. Only for validation/confirmation purposes, not necessary for generating explanations.
        if validate:
            rejected_validation(
                                dual_coef = dual_coef,
                                support_vectors = support_vectors,
                                intercept = intercept,
                                precision = precision,
                                t_lower = t_lower,
                                t_upper = t_upper,
                                lower_bound = lower_bound,
                                upper_bound = upper_bound,
                                data = data,
                                features_ranges = features_ranges,
                                show_log = show_log,
                                n_threads = n_threads
            )

        explanations.append(explanation)
    return explanations

def binary_validation(dual_coef, support_vectors, intercept, data, features_ranges, lower_bound = 0, upper_bound = 1, t_lower = 0, t_upper = 0,
                                      show_log = 0, n_threads = 1, precision = 0.0001, classified = "Positive", problem_name = "SVM_Explanation"):
    validate_prob = None
    X_validation = np.asarray([pulp.LpVariable('xv'+str(i+1), lowBound = lower_bound, upBound = upper_bound, cat='Continuous') for i in range(len(data[0]))])
    if classified == "Positive":
            validate_prob = pulp.LpProblem("Validation_of_Answer", pulp.LpMinimize)
            validate_prob += ((dual_coef @ support_vectors) @ X_validation.reshape(1, len(X_validation)).T + intercept)[0][0] <= t_lower -precision
            validate_prob += 1
    else:
            validate_prob = pulp.LpProblem("Validation_of_Answer", pulp.LpMaximize)
            validate_prob += ((dual_coef @ support_vectors) @ X_validation.reshape(1, len(X_validation)).T + intercept)[0][0] >= t_upper + precision
            validate_prob += 1
            
    for i, ranges in enumerate(features_ranges):
        X_validation[i].lowBound = ranges[0]
        X_validation[i].upBound = ranges[1]
    sat_validation = validate_prob.solve(PULP_CBC_CMD(msg=show_log, threads=n_threads, warmStart=True))
    if sat_validation == 1:
        print("Found intervals DO NOT maintain the class.")
        validation_values = []
        for v in X_validation:
            validation_values.append(v.varValue)
        print(validation_values)

    else:
        print(f"Found intervals MAINTAIN the class.")
    
    
def svm_explanation_binary(dual_coef, support_vectors, intercept, data, lower_bound = 0, upper_bound = 1, t_lower = 0, t_upper = 0, validate = False, show_log = 0, n_threads = 1,
                                       precision = 0.0001, classified = "Positive", problem_name = "SVM_Explanation"):
    """
    ## Generates explanations for rejected class.
    dual_coef - weight vector from trained SVC.
    support_vectors - support_vectors from trained SVC.
    intercept - intercept (bias) from trained SVC.
    data - patterns to be explained.
    lower_bound - lower bound of the features.
    upper_bound - upper bound of the features.
    t_lower - threshold of the negative class.
    t_upper - threshold of the positive class.
    validate - checks whether the generated explanations are valid.
    if not valid, then there is probably a problem with the given parameters.
    show_log - enables the pulp generated log.
    n_threads - enables the use of pulp multithreading.
    precision - enables bypassing the equality restriction of >= and <= by adding a small noise.
    classified - the class of the patterns to be explained. Positive or Negative.
    problem_name - the name of the LP problem.
    """
    #List of generated explanations
    explanations = []
    
    #Specify the types of Optimization Problems
    relevant_prob = pulp.LpProblem("Relevant_Features_"+problem_name, pulp.LpMinimize)
    
    #Specify that the features value range is between 0 and 1 (normalized dataset)
    X = np.asarray([pulp.LpVariable('x'+str(i+1), lowBound = lower_bound, upBound = upper_bound,cat='Continuous') for i in range(len(data[0]))])

    #Defining the threshold where the SVM decision function returns the Positive/Negative class.
    if classified == "Positive":
        relevant_prob += ((dual_coef @ support_vectors) @ X.reshape(1, len(X)).T + intercept)[0][0] <= t_upper - precision #All that are inferior to the upper threshold

    else:
        relevant_prob += ((dual_coef @ support_vectors) @ X.reshape(1, len(X)).T + intercept)[0][0] >= t_lower + precision #All that are superior to the lower threshold


    #For every sample
    for z in range(len(data)):
        relevant_features = []
        features_ranges = []
        not_relevant = []
        explanation = []
        
        #Setting up Prob Variables
        for x in X:
            x.lowBound = lower_bound
            x.upBound = upper_bound

        #For every feature    
        for j in range(len(data[z])):
            #The feature to be checked
            exclude = j
            #Iterate over every feature of the pattern
            for i, feature in enumerate(data[z]):
                #If feature is relevant, keep it so that it maintains the class
                if i != exclude and i in relevant_features:
                    X[i].setInitialValue(feature)
                    X[i].fixValue()

                #If its not the feature to be checked and haven't been worked upon yet
                elif i != exclude and i not in not_relevant and i not in relevant_features:
                    X[i].setInitialValue(feature)
                    X[i].fixValue()

                #If feature is the one to be checked or is irrelevant    
                elif i == exclude or i in not_relevant:
                    X[i].lowBound = lower_bound
                    X[i].upBound = upper_bound
            relevant_prob += X[exclude]
            
            #Feature value is originally upper/lower limited by the same value
            relevance_value = [data[z][exclude],data[z][exclude]]
            
            #Check if the feature is relevant and makes the predicted class change
            sat = relevant_prob.solve(PULP_CBC_CMD(msg=show_log, threads=n_threads, warmStart=True))
            if sat == 1:
                values = []
                for v in X:
                    values.append(v.varValue)
                relevant_features.append(exclude)
                features_ranges.append(relevance_value)
                explanation.append((exclude, data[z][exclude]))

            else:
                #Checked feature is not able to change the predicted class.
                not_relevant.append(exclude)
                features_ranges.append([0,1])
                
        #Validate explanations. Only for validation/confirmation purposes, not necessary for generating explanations.
        if validate:
            binary_validation(
                                dual_coef = dual_coef,
                                support_vectors = support_vectors,
                                intercept = intercept,
                                data = data,
                                precision = precision,
                                features_ranges = features_ranges,
                                lower_bound = lower_bound,
                                upper_bound = upper_bound,
                                t_lower = t_lower,
                                t_upper = t_upper,
                                classified = classified,
                                show_log = show_log,
                                n_threads = n_threads, 
            )

        
        explanations.append(explanation)
    return explanations