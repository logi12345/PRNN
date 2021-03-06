import numpy as np
import math
from DataProcessor import DataProcessor


class feed_forward_neurel_network:
    def __init__(self, hidden_layer_neurons):
        self.hidden_layer_neurons = hidden_layer_neurons

    def train(self, X, y, feature_space_size, output_size, iterations):
        # initialize weights
        W1 = np.random.randn(output_size, self.hidden_layer_neurons) / np.sqrt(2)
        W2 = np.random.randn(self.hidden_layer_neurons, feature_space_size) / np.sqrt(2)
        B1 = 0
        B2 = 0
        for i in range(iterations):
            h_net, h_out, o_net, o_out = self.forward_propogation(X, W1, W2, B1, B2)
            delta_W2 = self.back_prop_output_layer(o_out, y, h_out)
            delta_W1 = self.back_prop_hidden_layer(o_out, y, h_out, W2)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def cost(self, ):
        pass

    def back_prop_hidden_layer(self, o_out, y, h_out, W2):
        delta_error = o_out - y
        delta_o_out = o_out*(1-o_out)
        delta_error_net = delta_error*delta_o_out

        return

    def back_prop_output_layer(self, o_out, y, h_out):
        delta_error = o_out - y
        oo_out = 1 - o_out
        delta_o_out = o_out * oo_out
        delta_error_out = delta_error * delta_o_out
        h_out_tran = h_out.T
        delta_weight = h_out_tran.dot(delta_error_out)
        dw_2 = delta_weight / len(o_out)
        return dw_2

    def forward_propogation(self, X, W1, W2, B1, B2):
        h_net = X.dot(W1)
        h_out = np.vectorize(self.sigmoid)(h_net)
        o_net = h_out.dot(W2)
        o_out = np.vectorize(self.sigmoid)(o_net)
        return h_net, h_out, o_net, o_out


my_nn = feed_forward_neurel_network(3)

dp = DataProcessor()
feature_set, labels = dp.create_2d_moon_shape_data()

my_nn.train(feature_set, labels, 2, 2, 1)
