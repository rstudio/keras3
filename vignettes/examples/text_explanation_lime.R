#' This example shows how to use lime to explain text data.

library(readr)
library(dplyr)
library(keras)
library(tidyverse)

# Download and unzip data

activity_url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00461/drugLib_raw.zip"
temp <- tempfile()
download.file(activity_url, temp)
unzip(temp, "drugLibTest_raw.tsv")


# Read dataset

df <- read_delim('drugLibTest_raw.tsv',delim = '\t')
unlink(temp)

# Select only rating and text from the whole dataset

df = df %>% select(rating,commentsReview) %>% mutate(rating = if_else(rating >= 8, 0, 1))

# This is our text
text <- df$commentsReview

# And these are ratings given by customers
y_train <- df$rating


# text_tokenizer helps us to turn each word into integers. By selecting maximum number of features
# we also keep the most frequent words. Additionally, by default, all punctuation is removed.

max_features <- 1000
tokenizer <- text_tokenizer(num_words = max_features)

# Then, we need to fit the tokenizer object to our text data

tokenizer %>% fit_text_tokenizer(text)

# Via tokenizer object you can check word indices, word counts and other interesting properties.

tokenizer$word_counts 
tokenizer$word_index

# Finally, we can replace words in dataset with integers
text_seqs <- texts_to_sequences(tokenizer, text)

text_seqs %>% head(3)

# Define the parameters of the keras model

maxlen <- 15
batch_size <- 32
embedding_dims <- 50
filters <- 64
kernel_size <- 3
hidden_dims <- 50
epochs <- 15

# As a final step, restrict the maximum length of all sequences and create a matrix as input for model
x_train <- text_seqs %>% pad_sequences(maxlen = maxlen)

# Lets print the first 2 rows and see that max length of first 2 sequences equals to 15
x_train[1:2,]

# Create a model
model <- keras_model_sequential() %>% 
  layer_embedding(max_features, embedding_dims, input_length = maxlen) %>%
  layer_dropout(0.2) %>%
  layer_conv_1d(
    filters, kernel_size, 
    padding = "valid", activation = "relu", strides = 1
  ) %>%
  layer_global_max_pooling_1d() %>%
  layer_dense(hidden_dims) %>%
  layer_dropout(0.2) %>%
  layer_activation("relu") %>%
  layer_dense(1) %>%
  layer_activation("sigmoid")

# Compile
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)

# Run
hist <- model %>%
  fit(
    x_train,
    y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.1
  )

# Understanding lime for Keras Embedding Layers

# In order to explain a text with LIME, we should write a preprocess function
# which will help to turn words into integers. Therefore, above mentioned steps 
# (how to encode a text) should be repeated BUT within a function. 
# As we already have had a tokenizer object, we can apply the same object to train/test or a new text.

get_embedding_explanation <- function(text) {
  
  tokenizer %>% fit_text_tokenizer(text)
  
  text_to_seq <- texts_to_sequences(tokenizer, text)
  sentences <- text_to_seq %>% pad_sequences(maxlen = maxlen)
}


library(lime)

# Lets choose some text (3 rows) to explain
sentence_to_explain <- train_sentences$text[15:17]
sentence_to_explain

# You could notice that our input is just a plain text. Unlike tabular data, lime function 
# for text classification requires a preprocess fuction. Because it will help to convert a text to integers 
# with provided function. 
explainer <- lime(sentence_to_explain, model = model, preprocess = get_embedding_explanation)

# Get explanation for the first 10 words
explanation <- explain(sentence_to_explain, explainer, n_labels = 1, n_features = 10,n_permutations = 1e4)


# Different graphical ways to show the same information

plot_text_explanations(explanation)

plot_features(explanation)

interactive_text_explanations(explainer)

