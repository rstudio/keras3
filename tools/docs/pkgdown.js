
$(document).ready(function() {

  // turn functions section into ref-table
  $('#functions').find('table').attr('class', 'ref-index');
  
  // remove full s3 class from methods
  //$(".ref-index a:contains('tensorflow.contrib.keras')").remove();
  
  // are we in examples?
  var examples = window.location.href.match("/articles/examples/") !== null;
  if (examples) {
    $('.template-vignette').addClass('examples');
    var header = $(".page-header > h1");
    var script = header.text();
    var a = $("<a></a>");
    a.text(script);
    a.attr('href', "https://github.com/rstudio/keras/blob/master/vignettes/examples/" + script);
    header.empty().append(a);
  }
  
  // manage active state of menu based on current page
  var nav = $('ul.navbar-nav').children();
  var menuAnchor = null;
  var path = window.location.pathname;
  if (path.match('/reference/') !== null)
    menuAnchor = nav.eq(3);
  else if (path.match('/articles/examples/') !== null)
    menuAnchor = nav.eq(2);
  else if (path.match('/articles/') !== null)
    menuAnchor = nav.eq(1);
  else
    menuAnchor = nav.eq(0);
  
  // add active classes
  if (menuAnchor !== null)
    menuAnchor.addClass('active');
});
