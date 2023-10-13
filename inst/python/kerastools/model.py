

import os
import threading
import queue
import concurrent.futures

from keras import Model 


class RModel(Model):

  def __init__(self, name = None):
    super(RModel, self).__init__(name = name)
 
  def call(self, inputs, mask = None, **kwargs):
    return self._r_call(inputs, mask, **kwargs)


def as_generator (r_generator):

  q = queue.Queue(maxsize = 10)
  it = iter(r_generator)

  # this generator will simply take elements from the queue
  # until it's finished.
  def keras_generator ():
    while True:
      e = q.get()
      if e == '__FINISH__':
          break
      yield e

  def eval_loop ():
    try:
      el = next(it)
    except StopIteration:
      el = "__FINISH__"

    q.put(el)

  eval_loop()

  return keras_generator(), eval_loop
