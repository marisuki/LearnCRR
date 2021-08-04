# Learning Conditional Regression Rules



## Setup

```python
# requirements: 
python >= 3.7
scikit-learn
numpy

# setup through Makefile:
make
```



## Test

```python
# recurrence results 
python3 test/test_birdmap.py
python3 test/test_abalone.py


# learn conditional optimization of other models:
# need a fit, predict method of the model "reg", well-fitted to sklearn.linear_model
# follow the methods in test/test_birdmap.py or test_abalone.py,
# set up schema, database and init-params of functions 
# then use cond_regress.separation(.) to train conditional regressions.
```



