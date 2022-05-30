#' Train a Siamese MLP on pairs of digits from the MNIST dataset.
#' 
#' It follows Hadsell-et-al.'06 [1] by computing the Euclidean
#' distance on the output of the shared network and by optimizing the
#' contrastive loss (see paper for mode details).
#' 
#' [1] "Dimensionality Reduction by Learning an Invariant Mapping"
#'     https://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
#' 
#' Gets to 98.11% test accuracy after 20 epochs.
#' 3 seconds per epoch on a AMD Ryzen 7 PRO 4750U (CPU)
#' 


library(keras)

rms_distance <- function(inputs) {
    K <- keras::backend()
    x <- inputs[[1]]
    K$sqrt(K$maximum(K$mean(K$square(x), axis=1, keepdims=TRUE), K$epsilon()))
}

contrastive_loss <- function(y_true, y_pred) {
    # Contrastive loss from Hadsell-et-al.'06
    # https://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    K <- keras::backend()
    margin = 1
    margin_square = K$square(K$maximum(margin - (y_pred), 0))
    K$mean((1 - y_true) * K$square(y_pred) + (y_true) * margin_square)
}
    
create_pairs <- function(x, y) {
    # Positive and negative pair creation.
    # Alternates between positive and negative pairs.
    digit_indices <- tapply(1:length(y), y, list)
    y1 <- y
    y2 <- sapply(y, function(a) sample(0:9,1,prob=0.1+0.8*(0:9==a)))
    idx1 <- 1:nrow(x)
    idx2 <- sapply(as.character(y2), function(a) sample(digit_indices[[a]],1))
    is_same  <- 1*(y1==y2)
    list(pair1 = x[idx1,], pair2 = x[idx2,], y = is_same)
}

compute_accuracy <- function(predictions, labels) {
    # Compute classification accuracy with a fixed threshold on distances.
    mean(labels[predictions > 0.5])
}

# the data, shuffled and split between train and test sets
mnist   <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test  <- mnist$test$x
y_test  <- mnist$test$y
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test  <- array_reshape(x_test, c(nrow(x_test), 784))
x_train <- x_train / 255
x_test  <- x_test / 255

# create training+test positive and negative pairs
tr <- create_pairs(x_train, y_train)
te <- create_pairs(x_test,  y_test)

names(tr)

##----------------------------------------------------------------------
## Network definition
##----------------------------------------------------------------------

# input layers
input_dim = 784
input_1 <- layer_input(shape = c(input_dim))
input_2 <- layer_input(shape = c(input_dim))
 
# definition of the base network that will be shared
base_network <- keras_model_sequential() %>%
    layer_dense(units = 128, activation = 'relu') %>%    
    layer_dropout(rate = 0.1) %>% 
    layer_dense(units = 128, activation = 'relu') %>%    
    layer_dropout(rate = 0.1) %>% 
    layer_dense(units = 128, activation = 'relu')    

# because we re-use the same instance `base_network`, the weights of
# the network will be shared across the two branches
branch_1 <- base_network(input_1)
branch_2 <- base_network(input_2)

# merging layer
out <- layer_subtract(list(branch_1, branch_2)) %>%
    layer_dropout(rate = 0.1) %>% 
    layer_dense(units = 16, activation = 'relu') %>%
    layer_dense(1, activation = "sigmoid")    

# create and compile model
model <- keras_model(list(input_1, input_2), out)

#-----------------------------------------------------------
# train
#-----------------------------------------------------------

model %>% compile(
    optimizer = "rmsprop",
    #loss = "binary_crossentropy",
    loss = contrastive_loss,
    metrics = metric_binary_accuracy
)

history <- model %>% fit(
    list(tr$pair1, tr$pair2), tr$y,
    batch_size = 128, 
    epochs = 20, 
    validation_data = list(
        list(te$pair1, te$pair2),
        te$y
    )
)

plot(history)

#-----------------------------------------------------------
# compute final accuracy on training and test sets
#-----------------------------------------------------------

tr_pred <- predict(model, list(tr$pair1, tr$pair2))[,1]
tr_acc  <- compute_accuracy(tr_pred, tr$y)
te_pred <- predict(model, list(te$pair1, te$pair2))[,1]
te_acc  <- compute_accuracy(te_pred, te$y)

sprintf('* Accuracy on training set: %0.2f%%', (100 * tr_acc))
sprintf('* Accuracy on test set: %0.2f%%', (100 * te_acc))

#-----------------------------------------------------------
# show some plots
#-----------------------------------------------------------

par(mfrow=c(1,1))
vioplot::vioplot( te_pred ~ te$y )

i=3
visualizePair <- function(i) {
    image(rbind(matrix( te$pair1[i,],28,28)[,28:1], matrix( te$pair2[i,],28,28)[,28:1]))    
    title(paste("true:", te$y[i],"|  pred:", round(te_pred[i],5)))
}
par(mfrow=c(3,3))
lapply(1:9, visualizePair)

    

