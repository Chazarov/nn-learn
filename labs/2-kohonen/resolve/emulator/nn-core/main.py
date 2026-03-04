import numpy as np
import numpy.typing as npt





INPUT_LAYER_SIZE = 4
CLUSTERS_COUNT = 5


weights: npt.NDArray[np.float64] = np.random.rand(CLUSTERS_COUNT, INPUT_LAYER_SIZE)


def forward_propagation(weights: npt.NDArray[np.float64], input_vector:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 