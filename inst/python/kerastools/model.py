

import os

if (os.getenv('KERAS_IMPLEMENTATION', 'keras') == 'tensorflow'):
  from tensorflow.python.keras.engine import Model
else:
  from keras.engine import Model
 
 
class RModel(Model):

  def __init__(self, name = None):
    super(RModel, self).__init__(name = name)
 
  def call(self, inputs, mask = None):
    return self._r_call(inputs, mask)
