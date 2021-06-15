
def get_vocabulary (layer):
  vocab = layer.get_vocabulary()
  return([x.decode("UTF-8") for x in vocab])
