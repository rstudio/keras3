
import itertools

def dataset_generator(dataset, session):

  iter = dataset.make_one_shot_iterator()
  batch = iter.get_next()

  def gen():
    while 1:
      yield session.run(batch)

  return gen()


