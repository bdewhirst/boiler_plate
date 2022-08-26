# future work to do
last updated 2022-08-25

## for 9/1
* more on "eda.py"
* (evaluate based on status of active searches, etc.)

## recent priorities

### more on logistic regression
- statsmodels logistic regression
  - regularization w/ statsmodels
  - tuning
- crossvalidation & logistic regression
  - k-fold crossval from 4.2.2. https://www.kaggle.com/code/mnassrib/titanic-logistic-regression-with-python/notebook
- logistic regression with pyspark (with AWS)?
- explore automated recursive feature elimination (better fit for other techniques?)

### more on EDA
- boilerplate code for:
  - correlation matrices
  - small multiple plots of x0, x1,...xN _v._ y 
    - (1xN vertical column)
  - small multiples of x0,... vs x1... (etc.)
- collect more EDA scraps from kaggle and other places

### clean up/ best practices
- docstrings for all functions
- type hints
  - for native types
  - for everything else
* mypy (w/ PyCharm integration)
* logging
  * logging levels

## other things to add at some point
* regularization (ridge, lasso, and links to related theory)
* (re)learn the background on the fit critera I'm less familiar with/ I've used less frequently/ less often (e.g., as outputted in statsmodels I typically ignore)
* low positive rates "gotchas" (fraud rate, etc.)
* exhaustive review of measures of false positive, false negative
* exhaustive review of "information gain" metrics/framing

## misc.
* more on xgboost
  * (better fits of examples in general-- but this repo isn't currently for tuning iteration)
* try NLP
  * ID corpus (ENRON? something simpler/smaller...)
* pandas, play with date and memory, using categorical labels instead, etc.

----
# Hold off for now:

## data wrangling stuff (low priority)
### GCS
* get it working
#### w/i context of GCS:
*  throw it all into a pipeline class or something

### even more general
* cache data from database when retrieved...
  * vs PySpark/ lazy execution methods which don't move the data back and forth
* move sqlite constants into a central constants or config file
  * less value unless sqlite is more dynamic

## titanic:
*  throw it all into a pipeline class or something
  * related reading-- it may be popular, but I'm not sure it is "good"