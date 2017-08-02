// utility function to load history
function load_history(callback) {
  var request = new XMLHttpRequest();
  request.onreadystatechange = function() {
  if (request.readyState === 4) {
    if (request.status === 200 || request.status === 0) {
      var data = JSON.parse(request.responseText);
      if (callback) 
        callback(data);
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

function init_charts(data, update) {
  
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
  
  // create a C3 chart for each metric
  var c3_charts = [];
  for (var i = 0; i<metric_names.length; i++) {
    
    // get the metric 
    var metric = metric_names[i];
    
    // special y-axis treatment for accuracy (always 0 to 1)
    var y_axis = {};
    if (metric === 'acc') {
      y_axis.max = 1;
      y_axis.min = 0;
      y_axis.padding = {
        top: 0,
        bottom: 0
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
  
    // track chart
    c3_charts.push(chart);
  }
  
  // update all charts every second
  if (update) {
    var updateInterval = setInterval(function() {
      load_history(function(data) {
        
        // refresh each metric
        for (var i = 0; i<metric_names.length; i++) {
          var metric = metric_names[i];
          var chart = c3_charts[i];
          chart.load({
            columns: chart_columns(metric, data)
          });
          chart.flush();
        }
        
        // stop refreshing metrics when we have all epochs
        var first_metric = data.metrics[Object.keys(data.metrics)[0]];
        if (first_metric.length >= epochs)
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


// check for a history json payload
var historyJson = document.getElementById('history').innerHTML;

// no payload, initialize charts from history.json
if (historyJson === "%s") {
  
  // initialize charts
  load_history(function(data) {
    init_charts(data, true);
  });
  
} else {
  
  // initialize charts from embedded payload
  var data = JSON.parse(historyJson);
  init_charts(data, false);
  
}





