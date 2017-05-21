# This example demonstrates the use of fasttext for text classification
# 
# Based on Joulin et al's paper:
#   
#   Bags of Tricks for Efficient Text Classification
# https://arxiv.org/abs/1607.01759
# 
# Results on IMDB datasets with uni and bi-gram embeddings:
#   Uni-gram: 0.8813 test accuracy after 5 epochs. 8s/epoch on i7 cpu.
# Bi-gram : 0.9056 test accuracy after 5 epochs. 2s/epoch on GTx 980M gpu.
# 
library(keras)
library(purrr)


# Set parameters:
# ngram_range = 2 will add bi-grams features
ngram_range <- 1
max_features <- 20000
maxlen <- 400
batch_size <- 32
embedding_dims <- 50
epochs <- 5

imdb_data <- dataset_imdb(num_words = max_features)

print(length(imdb_data$train$x)) # train sequences
print(length(imdb_data$test$x)) # test sequences
print(sprintf("Average train sequence length: %f", mean(map_int(imdb_data$train$x, length))))
print(sprintf("Average test sequence length: %f", mean(map_int(imdb_data$test$x, length))))

