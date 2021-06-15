
import os

if (os.getenv('KERAS_IMPLEMENTATION', 'tensorflow') == 'keras'):
  from keras.engine.topology import Layer
  def shape_filter(shape):
    return shape
else:
  from tensorflow.keras.layers import Layer
  def shape_filter(shape):
    if isinstance(shape, tuple):
      return list(shape)
    elif not isinstance(shape, list):
      return shape.as_list()
    else:
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

    # call R to compute the output shape
    output_shape = self.r_compute_output_shape(shape_filter(input_shape))

    # if it was a list of lists then leave it alone, otherwise force to tuple
    # so that R users don't need to explicitly return a tuple
    if all(isinstance(x, (tuple,list)) for x in output_shape):
      return output_shape
    else:
      return tuple(output_shape)
