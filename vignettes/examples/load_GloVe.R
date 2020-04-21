# This example shows how one can quickly load glove vectors
# and train a Keras model in R

library(keras)
library(dplyr)

# Download Glove vectors if necessary
if (!file.exists('glove.6B.zip')) {
  download.file('http://nlp.stanford.edu/data/glove.6B.zip',destfile = 'glove.6B.zip')
  unzip('glove.6B.zip')
}

# load an example dataset from text2vec
library(text2vec)
data("movie_review")
as_tibble(movie_review)

# load glove vectors into R
vectors = data.table::fread('glove.6B.300d.txt', data.table = F,  encoding = 'UTF-8') 
colnames(vectors) = c('word',paste('dim',1:300,sep = '_'))

# structure of the vectors
as_tibble(vectors)

# define parameters of Keras model
library(keras)
max_words = 1e4
maxlen = 60
dim_size = 300

# tokenize the input data and then fit the created object
word_seqs = text_tokenizer(num_words = max_words) %>%
  fit_text_tokenizer(movie_review$review)

# apply tokenizer to the text and get indices instead of words
# later pad the sequence
x_train = texts_to_sequences(word_seqs, movie_review$review) %>%
  pad_sequences( maxlen = maxlen)

# extract the output
y_train = as.matrix(movie_review$sentiment)

# unlist word indices
word_indices = unlist(word_seqs$word_index)

# then place them into data.frame 
dic = data.frame(word = names(word_indices), key = word_indices, stringsAsFactors = FALSE) %>%
  arrange(key) %>% .[1:max_words,]

# join the words with GloVe vectors and
# if word does not exist in GloVe, then fill NA's with 0
word_embeds = dic  %>% left_join(vectors) %>% .[,3:302] %>% replace(., is.na(.), 0) %>% as.matrix()

# Use Keras Functional API 
input = layer_input(shape = list(maxlen), name = "input")

model = input %>%
  layer_embedding(input_dim = max_words, output_dim = dim_size, input_length = maxlen, 
                  # put weights into list and do not allow training
                  weights = list(word_embeds), trainable = FALSE) %>%
  layer_spatial_dropout_1d(rate = 0.2 ) %>%
  bidirectional(
    layer_gru(units = 80, return_sequences = TRUE) 
  )
max_pool = model %>% layer_global_max_pooling_1d()
ave_pool = model %>% layer_global_average_pooling_1d()

output = layer_concatenate(list(ave_pool, max_pool)) %>%
  layer_dense(units = 1, activation = "sigmoid")

model = keras_model(input, output)

# instead of accuracy we can use "AUC" metrics from "tensorflow.keras"
model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = tensorflow::tf$keras$metrics$AUC()
)

history = model %>% keras::fit(
  x_train, y_train,
  epochs = 8,
  batch_size = 32,
  validation_split = 0.2
)


