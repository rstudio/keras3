$(function() {
  $("#sidebar").stick_in_parent({offset_top: 40});
  $('body').scrollspy({
    target: '#sidebar',
    offset: 60
  });

});

$(document).ready(function() {
  
  
  // remove full s3 class from methods
  $(".ref-index a:contains('tensorflow.contrib.keras')").remove();
});

