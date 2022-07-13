# Documentation

## general notes
* all scripts are assumed to execute with the parent directory as their working directory (_i.e_, `> python3 setup/sqlite_setup.py`)

## Database Setup
1. edit the constants (all-caps) in setup/sqlite_setup.py to reflect your local machine and OS
2. download test.csv, train.csv, titanic.csv (etc.) into the /data subdirectory
3. run sqlite_setup.py

This is meant to represent a "business database" data would be sourced from in practice.