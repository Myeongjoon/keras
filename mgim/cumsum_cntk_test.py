from keras import backend
import numpy as np
import cntk as C

def cumsum(x, axis=0):
   dim = x.shape[axis]
   print('dim')
   print(dim)
   U = C.constant(np.triu(np.ones((dim, dim))).astype(x.dtype))
   print('U')
   print(U)
   if axis != -1:
       x = C.swapaxes(x, -1, axis)
       print('swapped')
       print(x())
   out = C.times(x, U)
   if axis != -1:
       out = C.swapaxes(out, -1, axis)
   return out

def parse_shape_or_val(shape_or_val):
    if isinstance(shape_or_val, np.ndarray):
        return shape_or_val.shape, shape_or_val
    else:
        return shape_or_val, np.random.random(shape_or_val).astype(np.float32) - 0.5

a,tensor = parse_shape_or_val((4,2))

print(tensor)
cumsum_res = cumsum(tensor)
print('result')
print(cumsum_res())