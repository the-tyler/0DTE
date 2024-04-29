# Quick Start

The quickest way to run the code in this repo is to use the following steps. 
Note: Currently, only pulls and saves data automatically


1) Open config.py 
2) Plug in the desired start and end date 
3) Save config.py
4) Open terminal

```
conda create -n finm python=3.12
conda activate finm
pip install -r requirements.txt
doit
```


## Generate signal of a skyrocketing 0DTE after 3PM  - upward trending market (October 30th 2023 - April 4th 2024)

- When S&P is in an upward trend, OTM call options are high in demand. OTM puts are also demanded by contrarians but it's a good sell. 

### Analysis
- Daily realized median vol and in (9:30-12:30), (9:30-2:00), (12:30-4:00), (2:30-4:00) of S&P
- Daily implied median vol and in (9:30-12:30), (9:30-2:00), (12:30-4:00), (2:30-4:00) of call options
- Daily Call Open Interest and in (9:30-12:30), (9:30-2:00), (12:30-4:00), (2:30-4:00) of call options
- Daily median bid-ask spread and in (9:30-12:30), (9:30-2:00), (12:30-4:00), (2:30-4:00) of call options
- Daily Flag which indicates if S&P's vol rose in (2:30-4:00)
- Daily Flag which indicates if there is important news feed tomorrow (we can rank the news feed in terms of strenghts between 0 and 1)
- Lag1 (daily) of realized and implied median vols 


### Modeling

- Because we are interested in the probability of 0DTEs rising, Logistic Regression is a suitable and easy-to-implement model
- The independent variables will be some of the analyzed variables above
- The dependent variable will be 0 if the 0DTEs expire worthless and 1 if their value rises (we need a systematic approach for mapping such as if the price of 0DTE rose minimum 30% after 2.30pm relative to its daily open)
- The model output will be 1 if the probability is greater than 50% and 0 otherwise. However, if the probability is much higher than 50% the position size can be increased and vice versa

# Project-Repo
# Project-Repo
