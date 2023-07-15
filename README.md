# Python forecasting demo

Project objective: create a 12 month forecast with random head count data that included ~23 years (282 months) of monthly history. Take the 12 month forecast and then back test it against the actual last 12 months of history to evaulate the accuracy. The specific methods used include a bidirectional LSTM model and a more traditional statistical based ARIMA model. 

## data description
- 282 months of random numbers representing head count for a fictional company
- start month is January 2000 and the ending month is June 2023
- mean head count = 773.7
- meadian head count = 774.5
- mode head count = 1491
- standard deviation head count = 391.13


  # read in the source data file


  
