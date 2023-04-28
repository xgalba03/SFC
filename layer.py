from neuron import Neuron


class Layer:
    """
        Layer is modeling a layer in the fully-connected-feedforward neural network architecture.
        It will play the role of connecting everything together inside and will be doing the backpropagation
        update.
    """

    def __init__(self, num_neuron, add_output_neuron=False):

        # List of neurons to be added to our layer
        self.neurons = []
        for i in range(num_neuron):
            # Create neuron - check if the layer is output layer so that we can add Delta later
            neuron = Neuron(i, is_output_neuron=add_output_neuron)
            self.neurons.append(neuron)

    def attach(self, layer):
        """
            This function attach the neurons from this layer to another one
            This is needed for the backpropagation algorithm
        """
        for in_neuron in self.neurons:
            in_neuron.attach_to_output(layer.neurons)

    def init_layer(self, num_input):
        """
            Iterate over each of the neuron and initialize
            the weights that connect with the previous layer
        """
        for neuron in self.neurons:
            neuron.init_weights(num_input)

    def predict(self, row):
        """
            Call the neuron predict function for each neuron
        """
        # First, we add the bias
        row.append(1)
        return [neuron.predict(row) for neuron in self.neurons]
