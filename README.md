## Stat 106 Final Project

This contains accompanying code for processing data in Python and models in R for our final project. 

`preprocess.py`: This takes ATP matches from the data directory to create `combined_atp_matches.csv`.

`create_features.py`: This takes `combined_atp_matches.csv` to create rolling features with a size 10 window. The output is `atp_matches_with_features.csv`.

`models.rmd`: This runs all of the milestone 4 models using `atp_matches_with_features.csv` as an input. The resulting output is captured in the knitted `models.pdf`

`ms2.ipynb`: Contains EDA done for milestone 2 as well as initial Elo modeling and random forest modeling with heavy feature engineering. Note that `models.rmd` ultimately contains the models we used for milestone 4.
