import pandas as pd
import numpy as np
import time
import sys

df = pd.read_csv("data/Small_data/CS170_Small_DataSet__87.txt", sep='\\s+')


def forward_selection():
    total_features = df.shape[1]-1
    current_set = []
    global_best = 0
    global_best_set = []

    default_rate = leave_one_out(df, [])
    print(f"Default rate: {default_rate}")
    
    for i in range(1,total_features+1):
        feature_to_add = 0
        best_so_far = 0
        print(f"On level {i} of the search tree")

        for j in range(1,total_features+1):
            if j in current_set:
                continue
            print(f"Considering adding the {j} feature")
            accuracy = leave_one_out(df, current_set+[j])
            print(f"Using feature {j} accuracy is {accuracy}%")

            if accuracy > best_so_far:
                best_so_far = accuracy
                feature_to_add = j
        current_set.append(feature_to_add)
        print(f"On level {i} added feature {feature_to_add}")
        if best_so_far > global_best:
            global_best = best_so_far
            global_best_set.append(feature_to_add)
    print(f"Global best set is {global_best_set}, global accuracy is {global_best}")
    return current_set, global_best

def leave_one_out(data, current_set):
    total_row = data.shape[0]
    correct_instance = 0
    data_col = data.values[:,current_set]
    labels = data.values[:,0]

    for i in range(total_row):
        testing = data_col[i,:]
        current_class = labels[i]
        nearest_dist = float('inf')
        nearest_label = None
        for j in range(total_row):
            if i == j: # not compare with itself
                continue
            dist = np.sqrt(np.sum((testing-data_col[j,:])**2)) # Euclid distance
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_label = labels[j]
        if nearest_label == current_class:
            correct_instance += 1
    return correct_instance/total_row
    
def backward_elimination():
    total_features = df.shape[1]-1
    current_set = list(range(1, total_features+1))
    print(f"the current_set has {current_set}")
    global_best = 0
    global_best_set = list(range(1, total_features+1))

    default_rate = leave_one_out(df, current_set)
    print(f"Default rate: {default_rate}")
    for i in range(1, total_features+1):
        feature_to_remove = None
        best_so_far = 0
        print(f"On level {i} of the search tree")
        for j in range(1, total_features+1):
            if j not in current_set:
                continue
            print(f"Considering removing the {j} feature")
            accuracy = leave_one_out(df, [f for f in current_set if f != j])
            print(f"Removing feature {j} accuracy is {accuracy}%")

            if accuracy > best_so_far:
                best_so_far = accuracy
                feature_to_remove = j
            
        print(f"On level {i} removed feature {feature_to_remove}")
        current_set.remove(feature_to_remove)
        print(f"Current_set is {current_set}")
        if best_so_far > global_best:
            global_best = best_so_far
            global_best_set = current_set
    print(f"Global best set is {global_best_set}, global accuracy is {global_best}")
    return current_set, global_best
    
def main():
    sys.stdout = open('output_log.txt', 'w')
    print(f"Classifying Sanity Dataset 1")
    option = input("Enter you choice: 1. Forward Selection 2. Backward Elimination")
    if option == "1":
        start_time = time.perf_counter()
        forward_selection()
        end_time = time.perf_counter()
        print(f"Total time spent: {end_time - start_time} seconds")
    elif option =="2":
        start_time = time.perf_counter()
        backward_elimination()
        end_time = time.perf_counter()
        print(f"Total time spent: {end_time - start_time} seconds")
    sys.stdout.close()


main()