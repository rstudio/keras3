
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

  // add google analytics
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){ (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o), m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m) })(window,document,'script','//www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-20375833-3', 'auto', {'allowLinker': true});
  ga('require', 'linker');
  ga('linker:autoLink', ['example-1.com'] );
  ga('send', 'pageview');
