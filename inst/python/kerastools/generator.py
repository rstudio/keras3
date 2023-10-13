import itertools
import types


def iterator_to_generator(iterator):
    """fit() and friends do specific checks for
    isinstance(x, types.GeneratorType, so a 'mere' iterator won't do"""

    def generator():
        yield from map(tuple, iterator)
        # yield from iterator

    return generator()


def dataset_generator(dataset, session):
    # used in TF v1 only
    iter = dataset.make_one_shot_iterator()
    batch = iter.get_next()

    def gen():
        while 1:
            yield session.run(batch)

    return gen()
