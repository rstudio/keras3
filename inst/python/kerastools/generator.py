
import itertools
import threading, queue, time
import concurrent.futures

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

is_finished = False
def fit_thread (model, generator, args):
  global is_finished
  is_finished = False
  q = queue.Queue(10)
  
  def event_loop(generator, future):
      global is_finished
      for element in generator:
          while True:
              try:
                  q.put(element, timeout = 0.01)
                  break
              except queue.Full:
                  if is_finished:
                      break
                  if future.done():
                      break
          if is_finished:
              break
          if future.done():
              break
      else:
          q.put('__FINISH__')
  
  def thread_fit():
    global is_finished
    def thread_gen():
      global is_finished
      while True:
        e = q.get()
        if e == '__FINISH__':
          break
        yield e
    output = model.fit(thread_gen(), **args)
    is_finished = True
    return output
    
  with concurrent.futures.ThreadPoolExecutor() as executor:
    future = executor.submit(thread_fit)
    event_loop(generator, future)
    out = future.result()
  
  return out
