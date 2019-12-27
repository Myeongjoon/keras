from keras import backend
import numpy as np



def parse_shape_or_val(shape_or_val):
    if isinstance(shape_or_val, np.ndarray):
        return shape_or_val.shape, shape_or_val
    else:
        return shape_or_val, np.random.random(shape_or_val).astype(np.float32) - 0.5

a,tensor = parse_shape_or_val((4,2))

print(backend.cumsum(tensor))