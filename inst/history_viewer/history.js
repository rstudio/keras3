// utility function to load history
function load_history(callback, on_error) {
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
  request.open('GET', "history.json");
  request.setRequestHeader('Cache-Control', 'no-cache');
  request.send(); 
}


// yield chart column data
function chart_columns(metric, data) {
  
  var total_epochs = data.params.epochs[0];
  var current_epochs = data.metrics[metric].length;
  var padding = [];
  for (var i = 0; i<(total_epochs-current_epochs); i++)
    padding.push(null);
  
  var columns = [
    [metric,null].concat(data.metrics[metric]).concat(padding)
  ];
  
  if (data.params.do_validation[0]) {
    var val_metric = "val_" + metric;
    columns.push(
      [val_metric,null].concat(data.metrics[val_metric]).concat(padding)
    );
  }
  
  return columns;
}

function init_charts() {
  
  // get the history json and parse it
  var historyJson = document.getElementById('history').innerHTML;
  var data = JSON.parse(historyJson);
  
  // alias params and metrics
  var params = data.params;
  var metrics = data.metrics;
  
  // determine metrics we will be plotting
  var metric_names = [];
  for (var i = 0; i<params.metrics.length; i++) {
    var metric = params.metrics[i];
    if (metric.lastIndexOf("val_", 0) !== 0)
      metric_names.push(metric);
  }
  
  // get the container and determine the initial height of charts
  var c3_container = document.getElementById("c3-container");
  var chart_height = c3_container.offsetHeight / metric_names.length;
  
  // helper function to tweak chart y-axis
  function adjust_y_axis(chart, metric, data) {
    var current_epochs = data.metrics[metric].length;
    if (metric === 'acc' && current_epochs > 0)
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
    if (metric === 'acc') {
      y_axis.padding = {
        top: 0
      };
    }
    
    // create a chart wrapper div
    var c3_div = document.createElement("div");
    c3_container.appendChild(c3_div);
    
    // create c3 chart bound to div
    var epochs = data.params.epochs[0];
    var tick_values = null;
    if (epochs <= 30) {
      tick_values = [];
      for (var n = 1; n <= epochs; n++)
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
        columns: chart_columns(metric, data)
      },
      size: {
        height: chart_height
      },
      transition: {
        duration: 20
      }
    });
    
    // adjust y axis
    adjust_y_axis(chart, metric, data);
  
    // track chart
    c3_charts.push(chart);

  }
  
  // helper to determine whether we've seen all the data
  function run_completed(data) {
    var epochs = data.params.epochs[0];
    var first_metric = data.metrics[Object.keys(data.metrics)[0]];
    return first_metric.length >= epochs;
  }
  
  // if the run isn't completed and we aren't runnign off of the filesystem
  // then update all charts every second
  if (!run_completed(data) && (window.location.protocol !== "file")) {
    var updateInterval = setInterval(function() {
      load_history(function(data) {
        
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







