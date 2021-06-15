
import itertools
import types

def iter_generator(iter):

  def gen():
    while 1:
      yield iter.next()

  return gen()

def dataset_generator(dataset, session):

  iter = dataset.make_one_shot_iterator()
  batch = iter.get_next()

  def gen():
    while 1:
      yield session.run(batch)

  return gen()
