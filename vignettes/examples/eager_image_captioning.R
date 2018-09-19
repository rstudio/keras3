#' This is the companion code to the post 
#' "Attention-based Image Captioning with Keras"
#' on the TensorFlow for R blog.
#' 
#' https://blogs.rstudio.com/tensorflow/posts/2018-09-17-eager-captioning

library(keras)
use_implementation("tensorflow")
library(tensorflow)
tfe_enable_eager_execution(device_policy = "silent")

np <- import("numpy")

library(tfdatasets)
library(purrr)
library(stringr)
library(glue)
library(rjson)
library(rlang)
library(dplyr)
library(magick)

maybecat <- function(context, x) {
  if (debugshapes) {
    name <- enexpr(x)
    dims <- paste0(dim(x), collapse = " ")
    cat(context, ": shape of ", name, ": ", dims, "\n", sep = "")
  }
}

debugshapes <- FALSE
restore_checkpoint <- FALSE
saved_features_exist <- FALSE

use_session_with_seed(7777,
                      disable_gpu = FALSE,
                      disable_parallel_cpu = FALSE)

annotation_file <- "train2014/annotations/captions_train2014.json"
image_path <- "train2014/train2014"

annotations <- fromJSON(file = annotation_file)

annot_captions <- annotations[[4]]
# 414113
num_captions <- length(annot_captions)

all_captions <- vector(mode = "list", length = num_captions)
all_img_names <- vector(mode = "list", length = num_captions)

for (i in seq_len(num_captions)) {
  caption <-
    paste0("<start> ", annot_captions[[i]][["caption"]], " <end>")
  image_id <- annot_captions[[i]][["image_id"]]
  full_coco_image_path <-
    sprintf("train2014/train2014/COCO_train2014_%012d.jpg", image_id)
  all_img_names[[i]] <- full_coco_image_path
  all_captions[[i]] <- caption
}

num_examples <- 30000

if (!saved_features_exist) {
  random_sample <- sample(1:num_captions, size = num_examples)
  train_indices <-
    sample(random_sample, size = length(random_sample) * 0.8)
  validation_indices <-
    setdiff(random_sample, train_indices)
  saveRDS(random_sample,
          paste0("random_sample_", num_examples, ".rds"))
  saveRDS(train_indices,
          paste0("train_indices_", num_examples, ".rds"))
  saveRDS(validation_indices,
          paste0("validation_indices_", num_examples, ".rds"))
} else {
  random_sample <-
    readRDS(paste0("random_sample_", num_examples, ".rds"))
  train_indices <-
    readRDS(paste0("train_indices_", num_examples, ".rds"))
  validation_indices <-
    readRDS(paste0("validation_indices_", num_examples, ".rds"))
}

sample_captions <- all_captions[random_sample]
sample_images <- all_img_names[random_sample]
train_captions <- all_captions[train_indices]
train_images <- all_img_names[train_indices]
validation_captions <- all_captions[validation_indices]
validation_images <- all_img_names[validation_indices]


load_image <- function(image_path) {
  img <- tf$read_file(image_path) %>%
    tf$image$decode_jpeg(channels = 3) %>%
    tf$image$resize_images(c(299L, 299L)) %>%
    tf$keras$applications$inception_v3$preprocess_input()
  list(img, image_path)
}


image_model <- application_inception_v3(include_top = FALSE,
                                        weights = "imagenet")

if (!saved_features_exist) {
  preencode <- unique(sample_images) %>% unlist() %>% sort()
  num_unique <- length(preencode)
  
  batch_size_4save <- 1
  image_dataset <- tensor_slices_dataset(preencode) %>%
    dataset_map(load_image) %>%
    dataset_batch(batch_size_4save)
  
  save_iter <- make_iterator_one_shot(image_dataset)
  save_count <- 0
  
  until_out_of_range({
    if (save_count %% 100 == 0) {
      cat("Saving feature:", save_count, "of", num_unique, "\n")
    }
    save_count <- save_count + batch_size_4save
    batch_4save <- save_iter$get_next()
    img <- batch_4save[[1]]
    path <- batch_4save[[2]]
    batch_features <- image_model(img)
    batch_features <- tf$reshape(batch_features,
                                 list(dim(batch_features)[1],-1L, dim(batch_features)[4]))
    for (i in 1:dim(batch_features)[1]) {
      p <- path[i]$numpy()$decode("utf-8")
      np$save(p,
              batch_features[i, ,]$numpy())
      
    }
    
  })
}

top_k <- 5000
tokenizer <- text_tokenizer(num_words = top_k,
                            oov_token = "<unk>",
                            filters = '!"#$%&()*+.,-/:;=?@[\\]^_`{|}~ ')
tokenizer$fit_on_texts(sample_captions)
train_captions_tokenized <-
  tokenizer %>% texts_to_sequences(train_captions)
validation_captions_tokenized <-
  tokenizer %>% texts_to_sequences(validation_captions)
tokenizer$word_index

tokenizer$word_index["<unk>"]

tokenizer$word_index["<pad>"] <- 0
tokenizer$word_index["<pad>"]

word_index_df <- data.frame(
  word = tokenizer$word_index %>% names(),
  index = tokenizer$word_index %>% unlist(use.names = FALSE),
  stringsAsFactors = FALSE
)

word_index_df <- word_index_df %>% arrange(index)

decode_caption <- function(text) {
  paste(map(text, function(number)
    word_index_df %>%
      filter(index == number) %>%
      select(word) %>%
      pull()),
    collapse = " ")
}

caption_lengths <-
  map(all_captions[1:num_examples], function(c)
    str_split(c, " ")[[1]] %>% length()) %>% unlist()
fivenum(caption_lengths)
max_length <- fivenum(caption_lengths)[5]

train_captions_padded <-
  pad_sequences(
    train_captions_tokenized,
    maxlen = max_length,
    padding = "post",
    truncating = "post"
  )
validation_captions_padded <-
  pad_sequences(
    validation_captions_tokenized,
    maxlen = max_length,
    padding = "post",
    truncating = "post"
  )

length(train_images)
dim(train_captions_padded)

batch_size <- 10
buffer_size <- num_examples
embedding_dim <- 256
gru_units <- 512
vocab_size <- top_k
features_shape <- 2048
attention_features_shape <- 64

train_images_4checking <- train_images[c(4, 10, 30)]
train_captions_4checking <- train_captions_padded[c(4, 10, 30),]
validation_images_4checking <- validation_images[c(7, 10, 12)]
validation_captions_4checking <-
  validation_captions_padded[c(7, 10, 12),]


map_func <- function(img_name, cap) {
  p <- paste0(img_name$decode("utf-8"), ".npy")
  img_tensor <- np$load(p)
  img_tensor <- tf$cast(img_tensor, tf$float32)
  list(img_tensor, cap)
}

train_dataset <-
  tensor_slices_dataset(list(train_images, train_captions_padded)) %>%
  dataset_map(function(item1, item2)
    tf$py_func(map_func, list(item1, item2), list(tf$float32, tf$int32))) %>%
  # dataset_shuffle(buffer_size) %>%
  dataset_batch(batch_size) 


cnn_encoder <-
  function(embedding_dim,
           name = NULL) {
    keras_model_custom(name = name, function(self) {
      self$fc <-
        layer_dense(units = embedding_dim, activation = "relu")
      
      function(x, mask = NULL) {
        # input shape: (batch_size, 64, features_shape)
        # shape after fc: (batch_size, 64, embedding_dim)
        maybecat("encoder input", x)
        x <- self$fc(x)
        maybecat("encoder output", x)
        x
      }
    })
  }

attention_module <-
  function(gru_units,
           name = NULL) {
    keras_model_custom(name = name, function(self) {
      self$W1 = layer_dense(units = gru_units)
      self$W2 = layer_dense(units = gru_units)
      self$V = layer_dense(units = 1)
      
      function(inputs, mask = NULL) {
        features <- inputs[[1]]
        hidden <- inputs[[2]]
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)
        # hidden shape == (batch_size, gru_units)
        # hidden_with_time_axis shape == (batch_size, 1, gru_units)
        hidden_with_time_axis <- k_expand_dims(hidden, axis = 2)
        
        maybecat("attention module", features)
        maybecat("attention module", hidden)
        maybecat("attention module", hidden_with_time_axis)
        
        # score shape == (batch_size, 64, 1)
        score <-
          self$V(k_tanh(self$W1(features) + self$W2(hidden_with_time_axis)))
        # attention_weights shape == (batch_size, 64, 1)
        attention_weights <- k_softmax(score, axis = 2)
        # context_vector shape after sum == (batch_size, embedding_dim)
        context_vector <-
          k_sum(attention_weights * features, axis = 2)
        
        maybecat("attention module", score)
        maybecat("attention module", attention_weights)
        maybecat("attention module", context_vector)
        
        list(context_vector, attention_weights)
      }
    })
  }

rnn_decoder <-
  function(embedding_dim,
           gru_units,
           vocab_size,
           name = NULL) {
    keras_model_custom(name = name, function(self) {
      self$gru_units <- gru_units
      self$embedding <-
        layer_embedding(input_dim = vocab_size, output_dim = embedding_dim)
      self$gru <- if (tf$test$is_gpu_available()) {
        layer_cudnn_gru(
          units = gru_units,
          return_sequences = TRUE,
          return_state = TRUE,
          recurrent_initializer = 'glorot_uniform'
        )
      } else {
        layer_gru(
          units = gru_units,
          return_sequences = TRUE,
          return_state = TRUE,
          recurrent_initializer = 'glorot_uniform'
        )
      }
      
      self$fc1 <- layer_dense(units = self$gru_units)
      self$fc2 <- layer_dense(units = vocab_size)
      
      self$attention <- attention_module(self$gru_units)
      
      function(inputs, mask = NULL) {
        x <- inputs[[1]]
        features <- inputs[[2]]
        hidden <- inputs[[3]]
        
        maybecat("decoder", x)
        maybecat("decoder", features)
        maybecat("decoder", hidden)
        
        c(context_vector, attention_weights) %<-% self$attention(list(features, hidden))
        
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x <- self$embedding(x)
        
        maybecat("decoder x after embedding", x)
        
        # x shape after concatenation == (batch_size, 1, 2 * embedding_dim)
        x <-
          k_concatenate(list(k_expand_dims(context_vector, 2), x))
        
        maybecat("decoder x after concat", x)
        
        # passing the concatenated vector to the GRU
        c(output, state) %<-% self$gru(x)
        
        maybecat("decoder output after gru", output)
        maybecat("decoder state after gru", state)
        
        # shape == (batch_size, 1, gru_units)
        x <- self$fc1(output)
        
        maybecat("decoder output after fc1", x)
        
        # x shape == (batch_size, gru_units)
        x <- k_reshape(x, c(-1, dim(x)[[3]]))
        
        maybecat("decoder output after reshape", x)
        
        # output shape == (batch_size, vocab_size)
        x <- self$fc2(x)
        
        maybecat("decoder output after fc2", x)
        
        list(x, state, attention_weights)
        
      }
    })
  }


encoder <- cnn_encoder(embedding_dim)
decoder <- rnn_decoder(embedding_dim, gru_units, vocab_size)

optimizer = tf$train$AdamOptimizer()

cx_loss <- function(y_true, y_pred) {
  mask <- 1 - k_cast(y_true == 0L, dtype = "float32")
  loss <-
    tf$nn$sparse_softmax_cross_entropy_with_logits(labels = y_true, logits =
                                                     y_pred) * mask
  tf$reduce_mean(loss)
}

get_caption <-
  function(image) {
    attention_matrix <-
      matrix(0, nrow = max_length, ncol = attention_features_shape)
    # shape=(1, 299, 299, 3)
    temp_input <- k_expand_dims(load_image(image)[[1]], 1)
    # shape=(1, 8, 8, 2048),
    img_tensor_val <- image_model(temp_input)
    # shape=(1, 64, 2048)
    img_tensor_val <- k_reshape(img_tensor_val,
                                list(dim(img_tensor_val)[1],-1, dim(img_tensor_val)[4]))
    # shape=(1, 64, 256)
    features <- encoder(img_tensor_val)
    
    dec_hidden <- k_zeros(c(1, gru_units))
    dec_input <-
      k_expand_dims(list(word_index_df[word_index_df$word == "<start>", "index"]))
    
    result <- ""
    
    for (t in seq_len(max_length - 1)) {
      c(preds, dec_hidden, attention_weights) %<-%
        decoder(list(dec_input, features, dec_hidden))
      attention_weights <- k_reshape(attention_weights, c(-1))
      attention_matrix[t, ] <- attention_weights %>% as.double()
      
      pred_idx = tf$multinomial(exp(preds), num_samples = 1)[1, 1] %>% as.double()
      
      pred_word <-
        word_index_df[word_index_df$index == pred_idx, "word"]
      
      if (pred_word == "<end>") {
        result <-
          paste(result, pred_word)
        attention_matrix <-
          attention_matrix[1:length(str_split(result, " ")[[1]]), , drop = FALSE]
        return (list(str_trim(result), attention_matrix))
      } else {
        result <-
          paste(result, pred_word)
        dec_input <- k_expand_dims(list(pred_idx))
      }
    }
    
    list(str_trim(result), attention_matrix)
  }

plot_attention <-
  function(attention_matrix,
           image_name,
           result,
           epoch) {
    image <-
      image_read(image_name) %>% image_scale("299x299!")
    result <- str_split(result, " ")[[1]] %>% as.list()
    # attention_matrix shape: nrow = max_length, ncol = attention_features_shape
    for (i in 1:length(result)) {
      att <- attention_matrix[i, ] %>% np$resize(tuple(8L, 8L))
      dim(att) <- c(8, 8, 1)
      att <- image_read(att) %>% image_scale("299x299") %>%
        image_annotate(
          result[[i]],
          gravity = "northeast",
          size = 20,
          color = "white",
          location = "+20+40"
        )
      overlay <-
        image_composite(att, image, operator = "blend", compose_args = "30")
      image_write(
        overlay,
        paste0(
          "attention_plot_epoch_",
          epoch,
          "_img_",
          image_name %>% basename() %>% str_sub(16,-5),
          "_word_",
          i,
          ".png"
        )
      )
    }
  }


check_sample_captions <-
  function(epoch, mode, plot_attention) {
    images <- switch(mode,
                     training = train_images_4checking,
                     validation = validation_images_4checking)
    captions <- switch(mode,
                       training = train_captions_4checking,
                       validation = validation_captions_4checking)
    cat("\n", "Sample checks on ", mode, " set:", "\n", sep = "")
    for (i in 1:length(images)) {
      c(result, attention_matrix) %<-% get_caption(images[[i]])
      real_caption <-
        decode_caption(captions[i,]) %>% str_remove_all(" <pad>")
      cat("\nReal caption:",  real_caption, "\n")
      cat("\nPredicted caption:", result, "\n")
      if (plot_attention)
        plot_attention(attention_matrix, images[[i]], result, epoch)
    }
    
  }

checkpoint_dir <- "./checkpoints_captions"
checkpoint_prefix <- file.path(checkpoint_dir, "ckpt")
checkpoint <-
  tf$train$Checkpoint(optimizer = optimizer,
                      encoder = encoder,
                      decoder = decoder)


if (restore_checkpoint) {
  checkpoint$restore(tf$train$latest_checkpoint(checkpoint_dir))
}

num_epochs <- 20

if (!restore_checkpoint) {
  for (epoch in seq_len(num_epochs)) {
    cat("Starting epoch:", epoch, "\n")
    total_loss <- 0
    progress <- 0
    train_iter <- make_iterator_one_shot(train_dataset)
    
    until_out_of_range({
      progress <- progress + 1
      if (progress %% 10 == 0)
        cat("-")
      
      batch <- iterator_get_next(train_iter)
      loss <- 0

      img_tensor <- batch[[1]]
      target_caption <- batch[[2]]
      
      dec_hidden <- k_zeros(c(batch_size, gru_units))
      
      dec_input <-
        k_expand_dims(rep(list(word_index_df[word_index_df$word == "<start>", "index"]), batch_size))
      
      with(tf$GradientTape() %as% tape, {
        features <- encoder(img_tensor)
        
        for (t in seq_len(dim(target_caption)[2] - 1)) {
          c(preds, dec_hidden, weights) %<-%
            decoder(list(dec_input, features, dec_hidden))
          loss <- loss + cx_loss(target_caption[, t], preds)
          dec_input <- k_expand_dims(target_caption[, t])
        }
        
      })
      total_loss <-
        total_loss + loss / k_cast_to_floatx(dim(target_caption)[2])
      
      variables <- c(encoder$variables, decoder$variables)
      gradients <- tape$gradient(loss, variables)
      
      optimizer$apply_gradients(purrr::transpose(list(gradients, variables)),
                                global_step = tf$train$get_or_create_global_step())
    })
    cat(paste0(
      "\n\nTotal loss (epoch): ",
      epoch,
      ": ",
      (total_loss / k_cast_to_floatx(buffer_size)) %>% as.double() %>% round(4),
      "\n"
    ))
    
    
    checkpoint$save(file_prefix = checkpoint_prefix)
    
    check_sample_captions(epoch, "training", plot_attention = FALSE)
    check_sample_captions(epoch, "validation", plot_attention = FALSE)
    
  }
}


epoch <- num_epochs
check_sample_captions(epoch, "training", plot_attention = TRUE)
check_sample_captions(epoch, "validation", plot_attention = TRUE)
