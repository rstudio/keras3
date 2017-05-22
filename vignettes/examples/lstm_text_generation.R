# Example script to generate text from Nietzsche's writings.
# 
# At least 20 epochs are required before the generated text
# starts sounding coherent.
# 
# It is recommended to run this script on GPU, as recurrent
# networks are quite computationally intensive.
# 
# If you try this script on new data, make sure your corpus
# has at least ~100k characters. ~1M is better.
library(keras)
