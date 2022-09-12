# Sundae Agenda 
2022-09-13 14:00 PST / 17:00 EST / 21:00 UTC
***
[Makesh Loganathan](https://www.linkedin.com/in/makesh-loganathan/)

[Brian Dewhirst](https://www.linkedin.com/in/brian-dewhirst-phd/)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
****
## (proposed) Agenda
* brief introductions
* quickly go over approach
  * PyCharm offers numerous advantages, especially longer-term
    * Brian is happy to support other users using other IDEs (etc.)
      * (if requested, show Brian can start jupyter notebook locally)
    * Today, local execution is planned
      * If highly desirable, Brian can replicate w/ AWS or GCP (with relaxed time pressure) as follow-up
    * In general, Brian will move from simple to complex
      * At each stage, I intend to have a workable fallback answer (likely from the previous stage)

* data sourcing
  * retrieve data
    * if data isn't in csv format, adjust accordingly
      * [native pandas data formats, methods](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html)
  * scope data size
    * n.b. 10x1000 csv of random floats is ~128 kb; anything under a gig should be fine
  * rename initial data as `/data/sundae-raw.csv`
    * If sampling approach will be taken, name and save that

* EDA
  * general overview of the data
    * size, columns
    * null vals
    * very wrong vals
    * plots/ small multiples
    * correlations
  * initial thoughts on approaches
  * iterative data exploration
    * data cleaning
    * initial thoughts on feature selection

* modeling
  * iterative model development
    * feature selection
    * model
    * score
    * sanity testing
    * iterate

* testing (as time permits)
  * unit tests
  * regression tests

***
