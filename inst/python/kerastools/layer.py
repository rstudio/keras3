
import os

if (os.getenv('KERAS_IMPLEMENTATION', 'tensorflow') == 'tensorflow'):
  from tensorflow.contrib.keras.python.keras import backend as K
  from tensorflow.contrib.keras.python.keras.engine.topology import Layer
else:
  from keras import backend as K
  from keras.engine.topology import Layer

class RLayer(Layer):

  def __init__(self, r_build, r_call, r_compute_output_shape, **kwargs):
    super(RLayer, self).__init__(**kwargs)
    self.r_build = r_build
    self.r_call = r_call
    self.r_compute_output_shape = r_compute_output_shape
    
  def build(self, input_shape):
    self.r_build(input_shape)
    super(RLayer, self).build(input_shape) 

  def call(self, x):
    return self.r_call(x)
      
  def compute_output_shape(self, input_shape):
    return self.r_compute_output_shape(input_shape)


