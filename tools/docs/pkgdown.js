
$(document).ready(function() {
  
  
  // turn functions section into ref-table
  $('#functions').find('table').attr('class', 'ref-index');
  
  // remove full s3 class from methods
  $(".ref-index a:contains('tensorflow.contrib.keras')").remove();
});

