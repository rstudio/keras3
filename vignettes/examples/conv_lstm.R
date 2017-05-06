# This script demonstrates the use of a convolutional LSTM network.
# This network is used to predict the next frame of an artificially
# generated movie which contains moving squares.
library(keras)


# Function Definition -----------------------------------------------------

generate_movies <- function(n_samples = 1200, n_frames = 15){
  
  rows <- 80
  cols <- 80
  
  noisy_movies <- array(0, dim = c(n_samples, n_frames, rows, cols))
  shifted_movies <- array(0, dim = c(n_samples, n_frames, rows, cols))
  
  n <- sample(3:8, 1)
  
  for(s in 1:n_samples){
    for(i in 1:n){
      # Initial position
      xstart <- sample(20:60, 1)
      ystart <- sample(20:60, 1)
      
      # Direction of motion
      directionx <- sample(-1:1, 1)
      directiony <- sample(-1:1, 1)
      
      # Size of the square
      w <- sample(2:3, 1)
      
      x_shift <- xstart + directionx*(0:(n_frames))
      y_shift <- ystart + directiony*(0:(n_frames))
      
      for(t in 1:n_frames){
        square_x <- (x_shift[t] - w):(x_shift[t] + w)
        square_y <- (y_shift[t] - w):(y_shift[t] + w)
        
        noisy_movies[s, t, square_x, square_y] <- 
          noisy_movies[s, t, square_x, square_y] + 1
        
        # Make it more robust by adding noise.
        # The idea is that if during inference,
        # the value of the pixel is not exactly one,
        # we need to train the network to be robust and still
        # consider it as a pixel belonging to a square.
        if(runif(1) > 0.5){
          noise_f <- sample(c(-1, 1), 1)
          
          square_x_n <- (x_shift[t] - w - 1):(x_shift[t] + w + 1)
          square_y_n <- (y_shift[t] - w - 1):(y_shift[t] + w + 1)
          
          noisy_movies[s, t, square_x_n, square_y_n] <- 
            noisy_movies[s, t, square_x_n, square_y_n] + noise_f*0.1
          
        }
        
        # Shift the ground truth by 1
        square_x_s <- (x_shift[t+1] - w):(x_shift[t+1] + w)
        square_y_s <- (y_shift[t+1] - w):(y_shift[t+1] + w)
        
        shifted_movies[s, t, square_x_s, square_y_s] <- 
          shifted_movies[s, t, square_x_s, square_y_s] + 1
      }
    }  
  }
  
  # Cut to a 40x40 window
  noisy_movies <- noisy_movies[,,21:60, 21:60]
  shifted_movies = shifted_movies[,,21:60, 21:60]
  
  noisy_movies[noisy_movies > 1] <- 1
  shifted_movies[shifted_movies > 1] <- 1

  
  list(
    noisy_movies = noisy_movies,
    shifted_movies = shifted_movies
  )
}





