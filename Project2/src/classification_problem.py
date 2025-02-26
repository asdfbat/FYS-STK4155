import numpy as np

class Classification():
    
    def __init__(self,hidden_activation='RELU',output_activation='softmax',cost_func='cross_entropy'):
        """
        Initializes a classification problem case to be used with class Neuralnetwork.
        Defining a cost function and activation functions for hidden and output layers,
        and defines the error from the output layer.
        
        Currently only one activation function may be chosen for all the hidden layers.
        
        Currently uses/available
        Cost functions:
        * cross_entropy
        Activation functions:
           Output layer:
        * softmax
           Hidden layers (applied to all):
        * sigmoid
        * RELU
        """
        self.h_a = hidden_activation
        self.o_a = output_activation
        self.cost = cost_func
     
    def hidden_activation(self,z,prime=False):
        """
        Returns the appropriate activation function for hidden layers given string from initialization.
        if prime = False returns the derivative
        """
        if self.h_a == 'RELU':
            if prime:
                return self._RELU_prime(z)
            else:
                return self._RELU(z)
        elif self.h_a == 'sigmoid':
            if prime:
                return self._sigmoid_prime(z)
            else:
                return self._sigmoid(z)
    
    def output_activation(self,z,prime=False):
        """
        Returns the appropriate activation function for the output layer given string from initialization.
        if prime = False returns the derivative
        """
        if self.o_a=='softmax':
            return self._softmax(z)
        
    def cost_function(self,a,t):
        """ Returns value of appropriate cost function """
        if self.cost == 'cross_entropy':
            return self._cross_entropy_cost(a,t)
    
    def output_error(self,a,t,z=None):
        """
        Computes the delta^L value, error from output layer
        using 
        * the gradient of cost function wrt a^L
        * the activation function of output layer (using softmax)
        Note z = None, not used here. Included for expansion to other functions
        which may depend on z too.
        """
        if self.cost == 'cross_entropy':
            return (a-t)
    
    
    def _sigmoid(self,z):
        """ Returns sigmoid activation function for hidden layers """
        return 1/(1+np.exp(-z))

    def _sigmoid_prime(self,z):
        """ Returns derivative of sigmoid """
        sigmoid = self._sigmoid(z)
        return sigmoid*(1-sigmoid)

    def _softmax(self,z):
        """ Returns the softmax function """
        exp_term = np.exp(z)
        return exp_term/np.sum(exp_term, axis=1, keepdims=True)
    
    def _RELU(self,z):
        """ Returns the RELU function """
        return np.where(z<0,0,z)

    def _RELU_prime(self,z):
        """ Returns the derivative of RELU """
        return np.where(z<0,0,1)
    
    def _cross_entropy_cost(self,a,t):
        """
        Computes cross entropy for an output a with target output t
        Uses np.nan_to_num to handle log(0) cases
        """
        return -np.sum(np.nan_to_num(t*np.log(a)-(1-t)*np.log(1-a)))


