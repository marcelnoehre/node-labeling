import numpy as np
import shapely

from shapely.geometry.base import BaseGeometry

def translate_many(geom: BaseGeometry, offsets: np.ndarray) -> np.ndarray:
    '''
    Copies of one geometry, each translated by its own offset.

    Parameters
    ----------
    geom : BaseGeometry
        geometry
    offsets : np.ndarray
        (n, 2) offsets

    Returns
    -------
    copies : np.ndarray
        n translated geometries
    '''
    offsets = np.asarray(offsets, dtype=float).reshape(-1, 2)
    k = shapely.get_num_coordinates(geom)
    shift = np.repeat(offsets, k, axis=0)
    return shapely.transform(np.full(len(offsets), geom, dtype=object), lambda c: c + shift)
