from nn_logic.training.test_backpropagation import test_bp_iteration, test_apply_adjustiments
from nn_logic.mathh.test_normalization import test_min_max_function, test_min_max_samples_normalize, test_min_max_signs_normalize


if __name__ == "__main__":
    test_bp_iteration()
    test_apply_adjustiments()
    test_min_max_signs_normalize()
    test_min_max_function()
    test_min_max_samples_normalize()