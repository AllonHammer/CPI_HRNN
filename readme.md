# Forecasting CPI Inflation with Hierarchical Recurrent Neural Networks
This Repository contains implementation of novel model we suggested in our paper.

1. Hierarchical GRU

In addition as a contribution for further research we provide he data set used in this work,
which is taken from the BLS after parsing and pre-processing. 

The data could be found on data/cpi_us_dataset.csv
note: this is a sample from the all data

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

@article{barkan2023forecasting,
  title={Forecasting CPI inflation components with hierarchical recurrent neural networks},
  author={Barkan, Oren and Benchimol, Jonathan and Caspi, Itamar and Cohen, Eliya and Hammer, Allon and Koenigstein, Noam},
  journal={International Journal of Forecasting},
  volume={39},
  number={3},
  pages={1145--1162},
  year={2023},
  publisher={Elsevier}
}
    pip install -r requirements.txt

    
To execute the code please run one of the following

1. hierarchical_gru.py

