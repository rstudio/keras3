
# monkey patch the keras.utils.Progbar class to have dynamic display
# (it currently detects dynamic display from a combination of 
# sys.stdout.isatty() and 'ipykernel' in sys.modules, neither one
# of which returns True within R.

import os

if (os.getenv('KERAS_IMPLEMENTATION', 'keras') == 'tensorflow'):
  from tensorflow.python.keras.utils import Progbar
else:
  from keras.utils import Progbar

def apply_patch():

  # save existing version of update for delegation
  update = Progbar.update
  
  def update_with_patch(self, current, values=None, force=False):
    # force dynamic display
    self._dynamic_display = True
    # delegate 
    update(self, current, values)
    
  # apply the patch
  Progbar.update = update_with_patch
