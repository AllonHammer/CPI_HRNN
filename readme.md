# Forecasting CPI Inflation with Hierarchical Recurrent Neural Networks
This Repository contains implementation of two novel models we suggested in our paper.

1. Hierarchical GRU
2. KNN Vectorial GRU

In addition as a contribution for further research we provide he data set used in this work,
which is taken from the BLS after parsing and pre-processing. 

The data could be found on resources/cpi_us_dataset.csv

## A brief description of the data set:
The data set contains the following columns:

* Date (differs by month)

* Category- item name

* Category id- item unique id

* Price- Seasonally adjusted CPI-U for the given month

* Weight- Relative importance of the item from the total aggregated index (=100)

*Indent- The hierarchy level (total aggregated index has indent 0, lowest level is 8)

*Parent- Parent’s item name

*Parent ID- Parent’s item ID


In order to run the code please make sure all Prerequisites are met (Pandas==0.22 in particular)
#### Prerequisites

    * Python >= 3.5
    * Pandas==0.22
    * tensorflow>=1.12.0
    * keras>=2.1.4 
    * numpy>=1.16.2
    * scipy>=1.2.1
    * scikit-learn>=0.19.1

    
To execute the code please run one of the following

1. hierarchical_gru.py
2. vec_gru.py

