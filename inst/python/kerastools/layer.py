
import os

if (os.getenv('KERAS_IMPLEMENTATION', 'keras') == 'tensorflow'):
  from tensorflow.python.keras._impl.keras.engine.topology import Layer
  def shape_filter(shape):
    if not isinstance(shape, list):
      return shape.as_list()
    else:
      return shape
else:
  from keras.engine.topology import Layer
  def shape_filter(shape):
    return shape

class RLayer(Layer):

  def __init__(self, r_build, r_call, r_compute_output_shape, **kwargs):
    super(RLayer, self).__init__(**kwargs)
    self.r_build = r_build
    self.r_call = r_call
    self.r_compute_output_shape = r_compute_output_shape
    
  def build(self, input_shape):
    self.r_build(shape_filter(input_shape))
    super(RLayer, self).build(input_shape) 

  def call(self, inputs, mask = None):
    return self.r_call(inputs, mask)
      
  def compute_output_shape(self, input_shape):
    return tuple(self.r_compute_output_shape(shape_filter(input_shape)))


