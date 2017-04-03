$(function() {
  $("#sidebar").stick_in_parent({offset_top: 40});
  $('body').scrollspy({
    target: '#sidebar',
    offset: 60
  });

});

$(document).ready(function() {
  
  
  // remove full s3 class from predict method
  $(".ref-index a:contains('predict.tensorflow.contrib.keras.python.keras.engine.training.Model')").remove();
});

