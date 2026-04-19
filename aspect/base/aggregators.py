"""Functions for collapsing dimensions."""

from typing import Callable, Union
from functools import partial

from numpy import asarray, nanmean, nanmedian, nansum, nanvar, ndarray, take
from numpy.linalg import norm
from numpy.typing import ArrayLike

AggFunction = Callable[[ArrayLike], ndarray]

AGGREGATORS = {
    "mean": nanmean,
    "median": nanmedian,
    "sum": nansum,
    "var": nanvar,
    "rms": partial(norm, ord=2, keepdims=False),
}


def slice_at_axis(
    a: ArrayLike,
    i: int, 
    axis: int = -1
):
    # n_dims = len(a.shape)
    # if axis < 0:
    #     axis = n_dims + axis
    # slicer = tuple([] if j == axis else [i] for j in range(n_dims))
    return take(a, asarray(i).astype(int), axis=axis)
    

def get_aggregator(
    aggregator: Union[str, AggFunction],
    axis: int = -1,
    **kwargs
) -> AggFunction:
    
    """Aggregate an array along an axis.

    Examples
    ========
    >>> import numpy as np
    >>> a = np.array([[3., 4.]])
    >>> rms = get_aggregator("rms", axis=-1)
    >>> norm2 = get_aggregator("2-norm", axis=-1)
    >>> rms(a)
    array([5.])
    >>> rms(a) == norm2(a)
    array([ True])
    >>> slicer_0 = get_aggregator(0, axis=-1) 
    >>> slicer_0(np.array([[1., 2.], [3., 4.]])) 
    array([1., 3.])

    """

    if isinstance(aggregator, int):
        return partial(slice_at_axis, i=aggregator, axis=axis)
    elif isinstance(aggregator, str):
        if aggregator.endswith("-norm") and aggregator.split("-")[0].isdigit():
            return partial(norm, ord=int(aggregator.split("-")[0]), axis=axis)
        elif aggregator in AGGREGATORS:
            return partial(AGGREGATORS[aggregator], axis=axis, **kwargs)
        else:
            raise ValueError(f"Invalid aggregator name '{aggregator}'!")
    elif isinstance(aggregator, Callable):
        return aggregator
    else:
        raise ValueError(f"Invalid aggregator type '{type(aggregator)}'! Must be a str or callable.")
