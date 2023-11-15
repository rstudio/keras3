Computes best sensitivity where specificity is >= specified value.

@description
`Sensitivity` measures the proportion of actual positives that are correctly
identified as such `(tp / (tp + fn))`.
`Specificity` measures the proportion of actual negatives that are correctly
identified as such `(tn / (tn + fp))`.

This metric creates four local variables, `true_positives`,
`true_negatives`, `false_positives` and `false_negatives` that are used to
compute the sensitivity at the given specificity. The threshold for the
given specificity value is computed and used to evaluate the corresponding
sensitivity.

If `sample_weight` is `NULL`, weights default to 1.
Use `sample_weight` of 0 to mask values.

If `class_id` is specified, we calculate precision by considering only the
entries in the batch for which `class_id` is above the threshold
predictions, and computing the fraction of them for which `class_id` is
indeed a correct label.

For additional information about specificity and sensitivity, see
[the following](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).

# Usage
Standalone usage:


```r
m <- metric_sensitivity_at_specificity(specificity = 0.5)
m$update_state(c(0, 0, 0, 1, 1), c(0, 0.3, 0.8, 0.3, 0.8))
m$result()
```

```
## [[ 60  61  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77
##    78  79  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95
##    96  97  98  99 100 101 102 103 104 105 106 107 108 109 110 111 112 113
##   114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131
##   132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149
##   150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167
##   168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185
##   186 187 188 189 190 191 192 193 194 195 196 197 198 199]]
```

```
## tf.Tensor(0.5, shape=(), dtype=float32)
```


```r
m$reset_state()
m$update_state(c(0, 0, 0, 1, 1), c(0, 0.3, 0.8, 0.3, 0.8),
               sample_weight = c(1, 1, 2, 2, 1))
m$result()
```

```
## [[ 60  61  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77
##    78  79  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95
##    96  97  98  99 100 101 102 103 104 105 106 107 108 109 110 111 112 113
##   114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131
##   132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149
##   150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167
##   168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185
##   186 187 188 189 190 191 192 193 194 195 196 197 198 199]]
```

```
## tf.Tensor(0.33333334, shape=(), dtype=float32)
```

Usage with `compile()` API:


```r
model %>% compile(
    optimizer = 'sgd',
    loss = 'mse',
    metrics = list(metric_sensitivity_at_specificity()))
```

@param specificity
A scalar value in range `[0, 1]`.

@param num_thresholds
(Optional) Defaults to 200. The number of thresholds to
use for matching the given specificity.

@param class_id
(Optional) Integer class ID for which we want binary metrics.
This must be in the half-open interval `[0, num_classes)`, where
`num_classes` is the last dimension of predictions.

@param name
(Optional) string name of the metric instance.

@param dtype
(Optional) data type of the metric result.

@param ...
Passed on to the Python callable

@export
@family confusion metrics
@family metrics
@seealso
+ <https:/keras.io/api/metrics/classification_metrics#sensitivityatspecificity-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/SensitivityAtSpecificity>

