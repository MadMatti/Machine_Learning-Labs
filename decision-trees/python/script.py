import monkdata as monk
import dtree
import random
from matplotlib import pyplot as plt
import numpy as np


# Assignment 1

'''Calculate the entropy of the three datasets'''
def entropy_calc():
    m1_entropy = dtree.entropy(monk.monk1)
    m2_entropy = dtree.entropy(monk.monk2)
    m3_entropy = dtree.entropy(monk.monk3)
    print("Entropy monk1 = " + str(m1_entropy) + "\n" + "Entropy monk2 = " + str(m2_entropy) + "\n" + "Entropy monk3 = " + str(m3_entropy))

# Assignment 3

'''Calculate the information gain of all the attributes of the three datasets'''
def gain_calc():
    m1_gain = []
    m2_gain = []
    m3_gain = []
    for i in range(6):
        m1_gain.append(dtree.averageGain(monk.monk1, monk.attributes[i]))
    for i in range(6):
        m2_gain.append(dtree.averageGain(monk.monk2, monk.attributes[i]))
    for i in range(6):
        m3_gain.append(dtree.averageGain(monk.monk3, monk.attributes[i]))
    print(m1_gain, m2_gain, m3_gain, sep="\n")

# Assignment 5

'''Build the three trees and print the errors'''
def build_tree(dataset):
    "Create a tree from the dataset passed as parameter"
    tree = dtree.buildTree(dataset, monk.attributes)
    return tree

def check_performace(tree, dataset):
    "Check the performaces of the passed tree against the passed dataset"
    return(dtree.check(tree, dataset))
    
def print_errors():
    print(1 - check_performace(build_tree(monk.monk1), monk.monk1))
    print(1 - check_performace(build_tree(monk.monk1), monk.monk1test))
    print(1 - check_performace(build_tree(monk.monk2), monk.monk2))
    print(1 - check_performace(build_tree(monk.monk2), monk.monk2test))
    print(1 - check_performace(build_tree(monk.monk3), monk.monk3))
    print(1 - check_performace(build_tree(monk.monk3), monk.monk3test))

# Assignment 7

'''Split the training data into a training set and a validation set'''
def splitData(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata)*fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

'''Prune the tree and print the best tree found'''
def prune(dataset, split, verbose=False):
    # split training data into training and validation
    train, valid = splitData(dataset, split)
    
    # build the tree
    tree = dtree.buildTree(train, monk.attributes)

    # print the tree
    if verbose:
        print(f"Reference tree: {tree}")

    # print the accuracy on the validation set
    reference_accuracy = dtree.check(tree, valid)
    if verbose:
        print(f"Accuracy on validation set: {reference_accuracy}\n")

    # prune the tree, find the best one, repeat until no more pruning is possible
    count=0
    best_pruned_accuracy = reference_accuracy
    while True:
        pruned_trees = dtree.allPruned(tree)
        # find the tree with the highest accuracy on the validation set
        best_tree = pruned_trees[0]
        best_current_pruned_accuracy = dtree.check(best_tree, valid)
        for tree in pruned_trees:
            pruned_accuracy = dtree.check(tree, valid)
            if pruned_accuracy > best_current_pruned_accuracy:
                best_tree = tree
                best_current_pruned_accuracy = pruned_accuracy
        
        if best_current_pruned_accuracy < best_pruned_accuracy:
            if verbose:
                print("No more pruning possible!\n\n")
            break

        best_pruned_accuracy = best_current_pruned_accuracy

        # print the tree
        if verbose:
            print(f"Best pruned tree: {tree}")

        # print the accuracy on the validation set
        if verbose:
            print(f"Accuracy on validation set: {best_pruned_accuracy}\n")

        count += 1

    return count, reference_accuracy, best_pruned_accuracy

'''Calculate and plot statistics for tree 1 and 3 considering different splitting parameters and 1000 runs'''
RUNS = 1000
def plot_stats():
    splits = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    data = {}
    
    for split in splits:
        data[split] = {
            "monk-1": [],
            "monk-3": [],
        }

        for i in range(RUNS):
            print(f"Run {i+1} for split {split}\n")
            print(f"Entropy of MONK-1: {dtree.entropy(monk.monk1)}\n")
            prunes, accuracy, prune_accuracy = prune(monk.monk1, split)
            data[split]["monk-1"].append({
                "prunes": prunes,
                "accuracy": accuracy,
                "prune_accuracy": prune_accuracy,
            })

            print(f"Entropy of MONK-3: {dtree.entropy(monk.monk3)}\n")
            prunes, accuracy, prune_accuracy = prune(monk.monk3, split)
            data[split]["monk-3"].append({
                "prunes": prunes,
                "accuracy": accuracy,
                "prune_accuracy": prune_accuracy,
            })

    # plot the data average and standard deviation
    monk1_avgs = []
    monk1_stds = []
    monk3_avgs = []
    monk3_stds = []
    for split in splits:
        monk1 = data[split]["monk-1"]
        monk3 = data[split]["monk-3"]

        monk1_prune_accuracy = [1 - x["prune_accuracy"] for x in monk1]
        monk1_avgs.append(np.mean(monk1_prune_accuracy))
        monk1_stds.append(np.std(monk1_prune_accuracy))

        monk3_prune_accuracy = [1 - x["prune_accuracy"] for x in monk3]
        monk3_avgs.append(np.mean(monk3_prune_accuracy))
        monk3_stds.append(np.std(monk3_prune_accuracy))

    # plot variance as shaded area
    plt.plot(splits, monk1_avgs, label="MONK-1")
    plt.fill_between(splits, np.array(monk1_avgs) - np.array(monk1_stds), np.array(monk1_avgs) + np.array(monk1_stds), alpha=0.2)
    plt.plot(splits, monk3_avgs, label="MONK-3")
    plt.fill_between(splits, np.array(monk3_avgs) - np.array(monk3_stds), np.array(monk3_avgs) + np.array(monk3_stds), alpha=0.2)

    plt.xlabel("Fraction")
    plt.ylabel("Error")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    entropy_calc()
    gain_calc()
    print_errors()
    plot_stats()