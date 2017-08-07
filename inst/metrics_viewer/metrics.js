// utility function to load metrics
function load_metrics(callback, on_error) {
  var request = new XMLHttpRequest();
  request.onreadystatechange = function() {
  if (request.readyState === 4) {
    if (request.status === 200 || request.status === 0) {
      try {
        var data = JSON.parse(request.responseText);
        if (callback) 
          callback(data);
      } catch(err) {
        if (on_error)
          on_error();
      }
    } else {
      if (on_error)
        on_error();
    }
  }
  };
  request.open('GET', "metrics.json");
  request.setRequestHeader('Cache-Control', 'no-cache');
  request.send(); 
}


// yield chart column data
function chart_columns(metric, data) {
  
  
  var columns = [
    [metric,null].concat(data[metric])
  ];
  
  var val_metric = "val_" + metric;
  if (data.hasOwnProperty(val_metric)) {
    columns.push(
      [val_metric,null].concat(data[val_metric])
    );
  }
  
  return columns;
}

function init_charts() {
  
  // get the metrics json and parse it
  var metricsJson = document.getElementById('metrics').innerHTML;
  var metrics = JSON.parse(metricsJson);
  
  // determine metrics we will be plotting (filter out val_ prefixed ones)
  var metric_names = [];
  var keys = Object.keys(metrics);
  for (var k = 0; k<keys.length; k++) {
    if (keys[k].lastIndexOf('val_', 0) !== 0)
      metric_names.push(keys[k]);
  } 

  // get the container and determine the initial height of charts
  var c3_container = document.getElementById("c3-container");
  var chart_height = c3_container.offsetHeight / metric_names.length;
  
  // helper function to see how many total epochs there are
  function get_total_epochs(data) {
    var first_metric = data[metric_names[0]];
    return first_metric.length;
  }
  
  // helper function to see how many epochs are in the data
  function get_current_epochs(data) {
    var first_metric = data[metric_names[0]];
    for (var r = 0; r<first_metric.length; r++) {
      if (first_metric[r] === null) {
        break;
      }
    }
    return r;
  }
  
  // helper function to determine whether a metric is 'accuracy'
  function is_accuracy(metric) {
    return metric === 'acc' || metric === 'accuracy';
  }
  
  // helper function to tweak chart y-axis
  function adjust_y_axis(chart, metric, data) {
    var current_epochs = get_current_epochs(data);
    if (is_accuracy(metric) && (current_epochs > 0))
      chart.axis.max({
        y: 1
      });
  }
  
  // helper to format y tick marks. use the default
  // d3 formatter but strip long sequences of zeros
  // followed by a single digit at the end (result
  // of JavaScript floating point rouning issues 
  // during axis interpolation)
  var default_format = d3.format("");
  function y_tick_format(d) {
    var fmt = default_format(d);
    return fmt.replace(/0+\d$/, '');
  }
  
  // create a C3 chart for each metric
  var total_epochs = get_total_epochs(metrics);
  var c3_charts = [];
  for (var i = 0; i<metric_names.length; i++) {
    
    // get the metric 
    var metric = metric_names[i];
    
    // default y_axis options
    var y_axis = {
      tick: {
        format: y_tick_format
      }
    };
    // special y-axis treatment for accuracy
    if (is_accuracy(metric)) {
      y_axis.padding = {
        top: 0
      };
    }
    
    // create a chart wrapper div
    var c3_div = document.createElement("div");
    c3_container.appendChild(c3_div);
    
    // create c3 chart bound to div
    var tick_values = null;
    if (total_epochs <= 30) {
      tick_values = [];
      for (var n = 1; n <= total_epochs; n++)
        tick_values.push(n);
    }
    var chart = c3.generate({
      bindto: c3_div,
      axis: {
        x: {
          min: 1,
          tick: {
            values: tick_values
          }
        },
        y: y_axis
      },
      data: {
        columns: chart_columns(metric, metrics)
      },
      size: {
        height: chart_height
      },
      transition: {
        duration: 20
      }
    });
    
    // adjust y axis
    adjust_y_axis(chart, metric, metrics);
  
    // track chart
    c3_charts.push(chart);

  }
  
  // helper to determine whether we've seen all the data
  function run_completed(data) {
    return get_current_epochs(data) >= get_total_epochs(data);
  }
  
  // if the run isn't completed and we aren't runnign off of the filesystem
  // then update all charts every second
  if (!run_completed(metrics) && (window.location.protocol !== "file")) {
    var updateInterval = setInterval(function() {
      load_metrics(function(data) {
        
        // refresh each metric
        for (var i = 0; i<metric_names.length; i++) {
          var metric = metric_names[i];
          var chart = c3_charts[i];
          chart.load({
            columns: chart_columns(metric, data)
          });
          adjust_y_axis(chart, metric, data);
          // ensure repaint
          chart.flush();
        }
        
        // stop refreshing metrics when the run is completed
        if (run_completed(data))
          clearInterval(updateInterval);
      },
      // error handler
      function() {
        clearInterval(updateInterval);
      });
    }, 1000);
  }
  
  // resize charts when window resizes
  window.addEventListener("resize", function(e) {
    var c3_container = document.getElementById("c3-container");
    var chart_height = c3_container.offsetHeight / c3_charts.length;
    for (var i = 0; i<c3_charts.length; i++) {
      var chart = c3_charts[i];
      var element = chart.element;
      element.style.maxHeight = "none";
      element.style.height = chart_height + "px";
      var elementRect = element.getBoundingClientRect();
      chart.resize({
        height: elementRect.height,
        width: elementRect.width
      });
    } 
  });  
}

// initialize charts
init_charts();







