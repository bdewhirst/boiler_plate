# future work 

## todo 2022-08-25
- tackle statsmodels logistic regression (unregularized) to start
- (reguarlized) logistic regression
  - and with scikit-learn instead of statsmodels?
- pyspark (w/i scope of this project, or aws stuff, or both?)
- boilerplate code for:
  - correlation matrices
  - small multiple plots of x0, x1,...xN _v._ y 
    - (1xN vertical column, I guess)



## focus areas:
* improve coverage of typing, docstrings, etc.
* more on xgboost
* try NLP
* pandas, play with date and memory, using categorical labels instead, etc.

### best practices stuff
* logging levels...
* mypy (w/ PyCharm integration)

## titanic:
* do additional simple (or not-so-simple) modeling
*  throw it all into a pipeline class or something

## GCS
* get it working

### w/i context of GCS:
*  throw it all into a pipeline class or something

## even more general
* cache data from database when retrieved...
* move sqlite constants into a central constants or config file (check, might be done)

## other things to consider
* regularization (ridge, lasso, and related theory)
* low positive rates "gotchas" (fraud rate, etc.)
* logistic regression/ techniques for binary determination
* measures of false positive, false negative