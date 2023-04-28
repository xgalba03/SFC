import sys
import random
import argparse
import csv

from perceptron import Perceptron


def get_args():
    """
        Parsing arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-mode", action="store", dest="mode",
                        choices=["train", "test", "xor_example"], default="train",
                        help="application mode")
    parser.add_argument("-model", action="store", dest="model",
                        help="name of file which contains already trained model")
    parser.add_argument("-show", action="store", dest="model",
                        choices=["yes", "no"], default="yes",
                        help="show the error while training the network")
    args_inner = parser.parse_args()

    if args_inner.mode == "test" and args_inner.model is None:
        print("Model was not selected.", file=sys.stderr)
        exit(-1)

    return args_inner


if __name__ == "__main__":
    args = get_args()

    with open('dataset.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    control = []
    del data[0]
    # Reformat the dataset features as explained in the documentation
    for current_list in data:
        del current_list[0]
        del current_list[5]
        del current_list[5]

        if current_list[0] == 'Male':
            current_list[0] = 1
        elif current_list[0] == 'Female':
            current_list[0] = 0
        else:
            current_list[0] = 2

        if float(current_list[1]) > 50:
            current_list[1] = 1
        else:
            current_list[1] = 0

        if current_list[4] == 'Yes':
            current_list[4] = 1
        elif current_list[4] == 'No':
            current_list[4] = 0
        else:
            current_list[4] = 2

        if float(current_list[5]) > 160:
            current_list[5] = 1
        else:
            current_list[5] = 0

        if current_list[6] == 'N/A':
            current_list[6] = 20
        elif float(current_list[6]) > 30:
            current_list[6] = 1
        else:
            current_list[6] = 0

        if current_list[7] == 'formerly smoked':
            current_list[7] = 1
        elif current_list[7] == 'never smoked':
            current_list[7] = 0
        elif current_list[7] == 'smokes':
            current_list[7] = 1
        else:
            current_list[7] = 0

        if current_list[8] == '1':
            current_list[8] = 1
        if current_list[8] == '0':
            current_list[8] == 0

        for i in range(len(current_list)):
            if current_list[i] == 'N/A':
                current_list[i] = 1
            current_list[i] = int(float(current_list[i]))

        control.append(current_list[8])
        del current_list[8]

    # Create a training dataset indexes
    # For the best results, train on 35% true cases and 65% false
    z_true = random.sample(range(1, 250), 125)
    z_false = random.sample(range(250, 4000), 175)

    # Create a testing dataset indexes
    test_true = random.sample(range(1, 250), 249)
    test_false = random.sample(range(250, 4000), 250)

    X_t = []
    Y_t = []
    X_test = []
    Y_test = []

    # Fill the values into test lists based on random indexes

    # Training data
    for m in z_true:
        X_t.insert(0, data[m])
        Y_t.insert(0, control[m])
    for n in z_false:
        X_t.insert(0, data[n])
        Y_t.insert(0, control[n])

    # Testing data
    for t_m in test_true:
        X_test.insert(0, data[t_m])
        Y_test.insert(0, control[t_m])
    for t_n in test_false:
        X_test.insert(0, data[t_n])
        Y_test.insert(0, control[t_n])

    original_data = data.copy()

    if args.mode == "train":
        clf = Perceptron(learning_rate=0.1, num_iteration=1500000)

        # Add layers
        clf.add_output_layer(num_neuron=1)
        clf.add_hidden_layer(num_neuron=5)
        clf.add_hidden_layer(num_neuron=9)

        # Train the network
        clf.train(X_t, Y_t)

        clf.save_model("model")
        print("[SUCCESSFUL RUN]")

    elif args.mode == "test":
        clf = Perceptron(learning_rate=0.1, num_iteration=3000000)
        clf.load_model(args.model)

        right = 0
        detected = 0
        false_positive = 0
        undetected = 0
        good_undetected = 0

        test_data = X_test
        control_data = Y_test

        for o in range(len(test_data)):
            prediction = clf.predict(test_data[o])
            actual = control_data[o]
            if actual == prediction:
                right += 1
            if actual == 1 and prediction == 1.0:
                detected += 1
            if actual == 0 and prediction == 1.0:
                false_positive += 1
            if actual == 1 and prediction == 0.0:
                undetected += 1
            if actual == 0 and prediction == 0.0:
                good_undetected += 1

        print("Accuracy: ", right / len(test_data))
        print("Number of detected strokes: ", detected)
        print("Detected no strokes ", good_undetected)
        print("Detected in percentage: ", detected / 250 * 100)
        print("False positives: ", false_positive)
        print("Undetected ", undetected)

    elif args.mode == "xor_example":
        # XOR function (one or the other but not both)
        X = [[0, 0, 0, 0],
             [0, 0, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 1, 1],
             [0, 1, 0, 0],
             [0, 1, 0, 1],
             [0, 1, 1, 0],
             [0, 1, 1, 1],
             [1, 0, 0, 0],
             [1, 0, 0, 1],
             [1, 0, 1, 0],
             [1, 0, 1, 1],
             [1, 1, 0, 0],
             [1, 1, 0, 1],
             [1, 1, 1, 0],
             [1, 1, 1, 1]]

        y = [0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

        clf = Perceptron(learning_rate=0.1, num_iteration=100000)
        clf.add_output_layer(num_neuron=1)
        clf.add_hidden_layer(num_neuron=3)
        clf.add_hidden_layer(num_neuron=2)

        # Train the network
        clf.train(X, y)

        print("Expected ", y[1], ", got: ", clf.predict(X[1]))
        print("Expected ", y[2], ", got: ", clf.predict(X[2]))
        print("Expected ", y[3], ", got: ", clf.predict(X[3]))
        print("Expected ", y[9], ", got: ", clf.predict(X[9]))
