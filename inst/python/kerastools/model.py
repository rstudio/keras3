

import os

if (os.getenv('KERAS_IMPLEMENTATION', 'tensorflow') == 'keras'):
  from keras.engine import Model
else:
  try:
    from tensorflow.python.keras.engine import Model
  except:
    from tensorflow.python.keras.engine.training import Model

class RModel(Model):

  def __init__(self, name = None):
    super(RModel, self).__init__(name = name)
 
  def call(self, inputs, mask = None, **kwargs):
    return self._r_call(inputs, mask, **kwargs)
