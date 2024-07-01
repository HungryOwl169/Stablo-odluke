import sys
from pathlib import Path
import csv
from collections import Counter
import math

file_path = Path(sys.argv[1])
lines = file_path.read_text().splitlines()
X = lines[0].strip().split(',')
label = X.pop()
depth = 1000000000000000000 

if (len(sys.argv) == 4):
    depth = int(sys.argv[3])
#print(depth)

with open(file_path, mode='r', newline='') as file:
    reader = csv.DictReader(file)
    D = [row for row in reader] 

file_path = Path(sys.argv[2])
lines = file_path.read_text().splitlines()
lines.pop(0)

with open(file_path, mode='r', newline='') as file:
    reader = csv.DictReader(file)
    D_test = [row for row in reader] 

""" for row in D_test:
    print(row) """

def entropy(D):
    label = next(reversed(D[0]))
    class_values = [entry[label] for entry in D]
    class_frequencies = Counter(class_values)
    #print(f'Class_frequencies: {class_frequencies}')
    return -sum((f/len(D)) * math.log2(f/len(D)) for f in class_frequencies.values())

def IG(D, x):
    x_values = set([entry[x] for entry in D])
    summ = 0
    for x_value in x_values:
        D_novi = [d for d in D if d[x] == x_value]
        #print(f'x-value = {x_value} D_novi = {D_novi}\n')
        summ = summ + (len(D_novi) / len(D)) * entropy(D_novi)
    return entropy(D) - summ

def filter_D(D, x, x_value):
    return [d for d in D if d[x] == x_value]

def argmax(D):
    label = next(reversed(D[0]))
    class_values = [entry[label] for entry in D]
    class_frequencies = Counter(class_values)
    max_freq = max(class_frequencies.values())
    keys_with_max_freq = [class_value for class_value, freq in class_frequencies.items() if freq == max_freq]
    keys_with_max_freq.sort()
    return keys_with_max_freq[0]
    
def argmaxIG(D, X):
    max_x = None
    max_ig = -1
    for x in X:
        ig = IG(D, x)
        if ig > max_ig:
            max_ig = ig
            max_x = x
        if ig == max_ig and x < max_x:
            max_ig = ig
            max_x = x
    return max_x

def are_lists_of_dicts_equal(list1, list2):
    tuple_list1 = [tuple(sorted(d.items())) for d in list1]
    tuple_list2 = [tuple(sorted(d.items())) for d in list2]

    return Counter(tuple_list1) == Counter(tuple_list2)

class Node:
    def __init__(self, is_leaf=False, x=None, v=None, prediction=None, D=None):
        self.x = x
        self.v = v
        self.prediction = prediction
        self.is_leaf = is_leaf 
        self.children = {}
        self.D = D


def id3(D, D_parrent, X, depth):
    if (len(D) == 0):
        prediction = argmax(D_parrent)
        return Node(is_leaf=True, prediction=prediction, D=D)
    v = argmax(D)
    label = list(D[0].keys())[-1]
    if (len(X) == 0 or are_lists_of_dicts_equal(D, filter_D(D, label, v))) or depth == 0:
        return Node(is_leaf=True, prediction=v, D=D)
    x = argmaxIG(D, X)
    subtrees = {}
    for v in set([i[x] for i in D]):
        Dx_v = [entry for entry in D if entry[x] == v]
        subtree = id3(Dx_v, D, [feature for feature in X if feature != x], depth = depth - 1)
        subtrees[v] = subtree
    
    root = Node(x=x, D=D)
    root.children = subtrees
    return root

tree = id3(D, D, X, depth)

def collect_branches(node, depth=1, prefix=""):
    branches = []
    if node.is_leaf:
        branches.append(prefix + node.prediction)
    else:
        for value, child in node.children.items():
            current_prefix = f"{prefix}{depth}:{node.x}={value} "
            branches.extend(collect_branches(child, depth + 1, current_prefix))
    return branches

def print_tree(tree):
    branches = collect_branches(tree)
    print("[BRANCHES]:")
    for branch in branches:
        print(branch)

def predict(tree, instance):
    node = tree
    while not node.is_leaf:
        value = instance.get(node.x)
        if value in node.children:
            node = node.children[value]
        else:
            return argmax(node.D)
    return node.prediction


print_tree(tree)

def print_predictions_accuracy_and_confusion_matrix(D_test):
    predictions = [predict(tree, d_test) for d_test in D_test]
    print('[PREDICTIONS]:', ' '.join(predictions))
    label = list(D_test[0].keys())[-1]
    correct_predictions = [entry[label] for entry in D_test]
    correct_prediction_counter = 0
    i = 0
    counters = {}
    for prediction in predictions:
        if (prediction == correct_predictions[i]):
            correct_prediction_counter = correct_prediction_counter + 1
        i = i + 1
    accuracy = correct_prediction_counter / len(correct_predictions)
    print(f'[ACCURACY]: {"{:.5f}".format(round(accuracy, 5))}')
    true_labels = [entry[label] for entry in D_test]
    unique_labels = sorted(set(true_labels))
    label_index = {label: idx for idx, label in enumerate(unique_labels)}
    
    cm = [[0] * len(unique_labels) for _ in range(len(unique_labels))]
    for true, pred in zip(true_labels, predictions):
        cm[label_index[true]][label_index[pred]] += 1
    
    print('[CONFUSION_MATRIX]:')
    for row in cm:
        print(' '.join(map(str, row)))

print_predictions_accuracy_and_confusion_matrix(D_test)


