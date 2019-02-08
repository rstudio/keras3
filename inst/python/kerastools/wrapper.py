import os

if (os.getenv('KERAS_IMPLEMENTATION', 'tensorflow') == 'keras'):
  from keras.layers import Wrapper
  def shape_filter(shape): 
    return shape
else:
  from tensorflow.python.keras.layers import Wrapper
  def shape_filter(shape):
    if not isinstance(shape, list):
      return shape.as_list()
    else:
      return shape
 
class RWrapper(Wrapper):

  def __init__(self, r_build, r_call, r_compute_output_shape, **kwargs):
    super(RWrapper, self).__init__(**kwargs)
    self.r_build = r_build
    self.r_call = r_call
    self.r_compute_output_shape = r_compute_output_shape
    
  def build(self, input_shape):
    self.r_build(shape_filter(input_shape))
    super(RWrapper, self).build(input_shape) 

  def call(self, inputs, mask = None):
    return self.r_call(inputs, mask)
      
  def compute_output_shape(self, input_shape):
    output_shape = self.r_compute_output_shape(shape_filter(input_shape))
    if all(isinstance(x, (tuple,list)) for x in output_shape):
      return output_shape
    else:
      return tuple(output_shape)


