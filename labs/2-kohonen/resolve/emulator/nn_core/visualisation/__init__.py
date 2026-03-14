from typing import List

import numpy as np
import numpy.typing as npt


def get_component_planes_visualisation(weights: npt.NDArray[np.float64], 
                                       input_names:List[str],
                                       input_id: int, # Определяет для какого инпута (id элемента входного вектора, ведь карта component_planes строится отдельно для каждого признака) нужно получить изображения component_planes
                                       samples: npt.NDArray[np.float64], 
                                       ) -> npt.NDArray[np.uint8]:
    """ Получение пикселей изображения. Отдельных карт для каждой входной переменной, показывающие её распределение по сетке."""


def get_u_matrix_visualisation(weights: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
    """Получение пикселей изображения u-матрицы для карты кохонена. В гексогональном виде"""
    