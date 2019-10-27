import numpy as np

class Classification():
    """
    Setting up cost function, and its derivative for logisitic regression problem.
    Also activation function for hidden and output layers
    Currently uses
    * CrossEntropy cost function
    * sigmoid activation for all hidden layers
    * softmax activation for output layer

    """
    @staticmethod
    def cross_entropy_cost(a,t):
        """
        Computes cross entropy for an output a with target output t
        Uses np.nan_to_num to handle log(0) cases
        """
        return -np.sum(np.nan_to_num(t*np.log(a)-(1-t)*np.log(1-a)))

    @staticmethod
    def output_error(a,t,z=None):
        """
        Computes the delta^L value, error from output layer
        using 
        * the gradient of cost function wrt a^L
        * the activation function of output layer (using softmax)
        Note z = None, not used here. Included for expansion to other functions
        which may depend on z too.
        """
        return (a-t)

