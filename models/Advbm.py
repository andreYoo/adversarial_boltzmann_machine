import torch
import numpy as np
import scipy
from torch.nn.functional import leaky_relu
from src.utils import sampling_bernoulli,sampling_gaussian
import pdb


class AdvBM():#Adversarial Boltzmann Machine
    def __init__(self, num_visible, num_hidden, k, learning_rate=1e-3, momentum_coefficient=0.5, weight_decay=1e-4, use_cuda=True):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.lmbd = 0.1
        self.weights = torch.randn(num_visible, num_hidden) * 0.1
        self.hidden_weights = torch.randn(num_hidden, num_hidden) * 0.1

        self.visible_bias = torch.ones(num_visible) * 0.5
        self.hidden_bias = torch.zeros(num_hidden)

        self.weights_momentum = torch.zeros(num_visible, num_hidden)
        self.hidden_weights_momentum = torch.zeros(num_hidden, num_hidden)
        self.visible_bias_momentum = torch.zeros(num_visible)
        self.hidden_bias_momentum = torch.zeros(num_hidden)

        if self.use_cuda:
            self.weights = self.weights.cuda()
            self.hidden_weights = self.hidden_weights.cuda()
            self.visible_bias = self.visible_bias.cuda()
            self.hidden_bias = self.hidden_bias.cuda()

            self.weights_momentum = self.weights_momentum.cuda()
            self.hidden_weights_momentum = self.hidden_weights_momentum.cuda()
            self.visible_bias_momentum = self.visible_bias_momentum.cuda()
            self.hidden_bias_momentum = self.hidden_bias_momentum.cuda()

    def sample_hidden(self, visible_probabilities):
        hidden_activations = torch.matmul(visible_probabilities, self.weights) + self.hidden_bias #VW+B
        hidden_probabilities = self._sigmoid(hidden_activations) #sigmoid (VW+B)
        return hidden_probabilities #output

    def sample_visible(self, hidden_probabilities):
        visible_activations = torch.matmul(torch.matmul(hidden_probabilities, self.hidden_weights),self.weights.t()) + self.visible_bias #hW_T+C
        visible_probabilities = leaky_relu(visible_activations) #Sigmoid (hW_T+c)
        return visible_probabilities #output

    def sample_hidden_to_hidden(self,hidden_probabilities):
        hidden_activations = torch.matmul(hidden_probabilities, self.hidden_weights) + self.hidden_bias  # hW_T+C
        hidden_probabilities = self._sigmoid(hidden_activations)  # Sigmoid (hW_T+c)
        return hidden_probabilities  # output

    def contrastive_divergence(self, input_data):
        # Positive phase for visible
        positive_hidden_probabilities = self.sample_hidden(input_data)  #up
        positive_hidden_activations = positive_hidden_probabilities
        #positive_hidden_activations = (positive_hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()  # Noise sampleing?
        positive_associations = torch.matmul(input_data.t(), positive_hidden_activations) #Back_down?

        # Negative phase for visible
        hidden_activations = positive_hidden_activations



        for step in range(self.k):
            visible_probabilities = self.sample_visible(hidden_activations)
            hidden_probabilities = self.sample_hidden(visible_probabilities)
            hidden_activations = hidden_probabilities
            #hidden_activations = (hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()

        negative_visible_probabilities = visible_probabilities
        negative_hidden_probabilities = hidden_probabilities
        negative_associations = torch.matmul(negative_visible_probabilities.t(), negative_hidden_probabilities)



        positive_visible_probabilities = self.sample_hidden_to_hidden(hidden_activations)
        positive_visible_activations  = positive_visible_probabilities
        positive_hidden_associations = torch.matmul(positive_visible_activations.t(), hidden_activations) #Back_down?


        batch_size = input_data.size(0)
        self.noise = torch.randn(batch_size, self.num_hidden) * 0.1
        self.noise = self.noise.cuda()
        for step in range(self.k):
            if step==0:
                visible_probabilities = self.sample_hidden_to_hidden(self.noise)
                hidden_probabilities = visible_probabilities
                hidden_activations = hidden_probabilities
            else:
                visible_probabilities = self.sample_hidden_to_hidden(hidden_activations)
                hidden_probabilities = visible_probabilities
                hidden_activations = hidden_probabilities
            #hidden_activations = (hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()

        tmp_negative_visible_probabilities = visible_probabilities
        tmp_negative_hidden_probabilities = hidden_probabilities

        negative_hidden_associations = torch.matmul(tmp_negative_visible_probabilities.t(), tmp_negative_hidden_probabilities)



        # Update parameters
        self.weights_momentum *= self.momentum_coefficient
        self.weights_momentum += (positive_associations - negative_associations)

        self.hidden_weights_momentum *= self.momentum_coefficient
        self.hidden_weights_momentum += (positive_hidden_associations - negative_hidden_associations)



        self.visible_bias_momentum *= self.momentum_coefficient
        self.visible_bias_momentum += torch.sum(input_data - negative_visible_probabilities, dim=0)

        self.hidden_bias_momentum *= self.momentum_coefficient
        self.hidden_bias_momentum += torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0)



        self.weights += self.weights_momentum * self.learning_rate / batch_size
        self.visible_bias += self.visible_bias_momentum * self.learning_rate / batch_size
        self.hidden_bias += self.hidden_bias_momentum * self.learning_rate / batch_size

        self.weights -= self.weights * self.weight_decay  # L2 weight decay
        self.hidden_weights -= self.hidden_weights * self.weight_decay  # L2 weight decay

        # Compute reconstruction error
        error = torch.sum((input_data - negative_visible_probabilities)**2)

        return error

    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def _random_probabilities(self, num):
        random_probabilities = torch.rand(num)

        if self.use_cuda:
            random_probabilities = random_probabilities.cuda()

        return random_probabilities

    def output(self, visible_probabilities):
        hidden_activations = torch.matmul(visible_probabilities, self.weights) + self.hidden_bias #VW+B
        hidden_probabilities = self._sigmoid(hidden_activations) #sigmoid (VW+B)
        return hidden_probabilities #output

    def generation(self):
        noise = torch.randn(64,128) * 0.1
        noise = noise.cuda()
        visible_activations = torch.matmul(torch.matmul(noise, self.hidden_weights),self.weights.t()) + self.visible_bias #hW_T+C
        visible_probabilities = leaky_relu(visible_activations) #Sigmoid (hW_T+c)
        return visible_probabilities #output