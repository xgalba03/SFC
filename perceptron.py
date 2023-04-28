import pickle
import random

from layer import Layer


class Perceptron:
    """
        Our perceptron will have 2 hidden layers:
            an input layer, a perceptron layer and a one neuron output layer which does binary classification
    """

    def __init__(self, learning_rate=0.01, num_iteration=100):

        # Initialize values
        self.layers = []
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration

    def add_output_layer(self, num_neuron):
        """
            # Initialize and add a new OUTPUT layer
        """
        output_layer = Layer(num_neuron, add_output_neuron=True)
        self.layers.insert(0, output_layer)

    def add_hidden_layer(self, num_neuron):
        """
            # Initialize and add a new HIDDEN layer, this time inserting to the front of the architecture
        """
        # Create a hidden layer
        hidden_layer = Layer(num_neuron)
        # Attach the last added layer to this new layer
        hidden_layer.attach(self.layers[0])
        # Add this layers to the architecture
        self.layers.insert(0, hidden_layer)

    def update_layers(self, target):
        """
            Iteratively update all the layers
                Calculate update weights in first iteration
                Update all the weights in the next one
        """
        # First iteration - using reverse order
        for layer in reversed(self.layers):

            # Actual calculation of neuron weights
            for neuron in layer.neurons:
                neuron.calculate_update(self.learning_rate, target)

        # Second iteration, using calculated weights to update the layers
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.update_neuron()

    def train(self, x, y):
        """
            Stochastic descend using backpropagation - randomly selecting a row from our training dataset and try to
            predict it's output value (the truth). Based on the correctness of the prediction, calculate the error
            and backpropagate it to calculate new weights
        """

        # Get the number of rows and features (there needs to be a feature for every row!)
        num_row = len(x)
        num_feature = len(x[0])

        # Init the layers with base features (number of neurons)
        self.layers[0].init_layer(num_feature)

        for i in range(1, len(self.layers)):
            num_input = len(self.layers[i - 1].neurons)
            self.layers[i].init_layer(num_input)

        # Launch the training algorithm
        for i in range(self.num_iteration):

            # Stochastic Gradient Descent
            r_i = random.randint(0, num_row - 1)
            row = x[r_i]  # take the random sample from the dataset
            self.predict(row)
            target = y[r_i]

            # Update the layers using backpropagation
            self.update_layers(target)

            # At every 1000 iteration we update and show the error
            if i % 1000 == 0:
                total_error = 0
                for r_i in range(num_row):
                    row = x[r_i]
                    yhat = self.predict(row)
                    error = y[r_i] - yhat
                    total_error = total_error + error ** 2
                mean_error = total_error / num_row
                print(f"Iteration {i} with error = {mean_error}, total: {total_error}")

    def predict(self, row):
        """
            Prediction function that will take a row of input and give back the output
            of the whole neural network.
        """

        # Gather all the activation in the hidden layer

        activations = self.layers[0].predict(row)
        for i in range(1, len(self.layers)):
            activations = self.layers[i].predict(activations)

        outputs = []
        for activation in activations:
            # Decide if we output a 1 or 0
            if activation >= 0.5:
                outputs.append(1.0)
            else:
                outputs.append(0.0)

        # We currently have only One output allowed
        return outputs[0]

    def save_model(self, name):
        """
            Saves model
        """
        with open("{}.pkl" .format(name), "wb") as model:
            pickle.dump(self, model, pickle.HIGHEST_PROTOCOL)

    def load_model(self, name):
        """
            Loads model
        """
        with open("{}.pkl".format(name), "rb") as model:
            tmp_model = pickle.load(model)

        self.layers = tmp_model.layers
        self.learning_rate = tmp_model.learning_rate
