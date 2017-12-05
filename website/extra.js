
$(document).ready(function() {

  // turn functions section into ref-table
  $('#functions').find('table').attr('class', 'ref-index');
  
  // are we in examples?
  var examples = window.location.href.match("/articles/examples/") !== null;
  if (examples) {
    $('.template-vignette').addClass('examples');
   
    // remove right column
    $(".col-md-9").removeClass("col-md-9").addClass('col-md-10');
    $(".col-md-3").remove();
  }
  
   // add search box
  var search_form = $.parseHTML('<input type="text" class="st-default-search-input" placeholder="Search">');
  var navbar_right = $('.navbar-right');
  navbar_right.append(search_form);
  
  // add swiftype
  (function(w,d,t,u,n,s,e){w['SwiftypeObject']=n;w[n]=w[n]||function(){
  (w[n].q=w[n].q||[]).push(arguments);};s=d.createElement(t);
  e=d.getElementsByTagName(t)[0];s.async=1;s.src=u;e.parentNode.insertBefore(s,e);
  })(window,document,'script','//s.swiftypecdn.com/install/v2/st.js','_st');
  _st('install','RzoPBEQeaoi3AAmD3UEK','2.0.0');
});
