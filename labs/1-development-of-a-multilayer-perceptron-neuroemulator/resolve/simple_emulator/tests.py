from training.test_backpropagation import test_backpropagation_iteration
from forwrdpropagation.test_forward_propagation import test_forward_propagation
from visualisation.test_visualisation import test_visualisation_iris



if __name__ == "__main__":
    test_forward_propagation()
    test_backpropagation_iteration()
    test_visualisation_iris()