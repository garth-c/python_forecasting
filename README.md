# Python forecasting demo

Project objective: create a 12 month forecast with random head count data that included ~23 years (282 months) of monthly history. Take the 12 month forecast and then back test it against the actual last 12 months of history to evaulate the accuracy. The specific methods used include a bidirectional LSTM model. For the bidirectional model, use TensorFlow as the back end tensor infrastructure. Also, this demo will use a time series generator to ingest the source data into batches and then feed it to the TensorFlow model for processing.

## data description
- 282 months of random numbers representing head count for a fictional company
- start month is January 2000 and the ending month is June 2023
- mean head count = 773.7
- meadian head count = 774.5
- mode head count = 1,491
- standard deviation head count = 391.13

---------------------------------------------------------------------------------

## The roadmap for this demo is below
- set up the computing environement in Pycharm
- read in the source data file
- prepare the source data for processing
- set up the time series generators
- build the model
- evaluate the model losses
- create a forecast training loop
- backtest the predictions 

---------------------------------------------------------------------------------

# set up the computing environement in Pycharm

The first thing to do is to set up the computing environment in the Pycharm IDE. This set up will set up the code to process properly according to my standards. I make use of setting the random seed number for certain key libraries like TensorFlow and Numpy. Also specific key functions from the Keras library are imported and these will be used in almost all aspects of this demo.

```
#set the random seed
import random as random
random_seed_number = 12345
random.seed(random_seed_number)

#set the working directory
import os as os
os.chdir('C:\\Users\\matri\\Documents\\my_documents\\local_git_folder\\python_code\\time_series_forecasting')
os.getcwd()

#tensor processing
import tensorflow as tf
print(tf.__version__)
tf.random.set_seed(random_seed_number)

#matrix math
import numpy as np
print(np.__version__)
np.random.seed(random_seed_number)

#data wrangling
import pandas as pd
print(pd.__version__)

#tensor flow dependency functions
from keras import layers
from keras.utils import Sequence
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, Bidirectional
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import EarlyStopping

#data preprocessing dependencies
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#plotting
import matplotlib.pyplot as plt
```

-------------------------------------------------------------------------------------------

# read in the source data file

The first operation step in this process is to read in this the source data from Excel and put it into a Pandas data frame. Then the data in the data frame is validated against the source data to make sure that it matches. 

```
###~~~
#read in the source data
###~~~

#source data
headcounts = pd.read_excel('head_counts.xlsx',
                            sheet_name = 'heads',
                            parse_dates = True,
                            index_col = 'date')

#get df info
headcounts.info()
```

The input file info details are below. Valdiate them against the input source data file.

<img width="241" alt="image" src="https://github.com/garth-c/python_forecasting/assets/138831938/cf610024-fff7-4a1a-b546-4e225aa12c33">


Next, produce a line plot to see the landscape of the data set. This plot gives a high level view of the overall direction and any themes that are obvious in the source data. This is a key step and it will inform other aspects of the coding.

![line_plot](https://github.com/garth-c/python_forecasting/assets/138831938/eb4d5701-5dca-4138-8ba0-a121a58db475)

------------------------------------------------------------------------------------------------------------------------

# prepare the source data for processing

Next the source data in the Pandas data frame need to be prepared for consumption by the model. This includes splitting the source data into training and testing sets in time series order as well as scaling the source data to normalize the inputs. The test set will consist of the last 24 months of data in this series and the train set will consist of all of the prior data in the series. With time series data, preserving the ordering of the data is vitally important to getting valid forecast values. In a non time series model, the assignment of data between training and testing would be random since order would not matter for those models. 

```
###~~~
#data prep section
###~~~

###test/validate/train split
#split into training and test datasets.
#since it's time series we should do it by date.

#set indexing for train test split
test_size = 24 #use 24 months as the test size cut off
test_ind = len(headcounts) - test_size

#train test split
train = headcounts.iloc[:test_ind]
test = headcounts.iloc[test_ind:]

#LSTM's in Keras require a 3D tensor with shape
#1)number of samples
#2)time steps
#3)number of predictor features

#Samples -> One sequence is one sample. A batch is composed of one or more samples
#Time Steps -> One time step is one point of observation in the sample
#Features -> One feature is one observation at a time step

#instantiate the scaler function
scaler = MinMaxScaler()

#scale only the training data
scaler.fit(train)

#make copies of the train / test data
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)
```
----------------------------------------------------------------------------------------------------------------------

# set up the time series generators

The time series generators for training and validation ingest the source data set and then carve it up into batches. From there the generators push each batch into model for processing. These generators are an invaluable way to process source data and feed it to a time series model in a controlled way. Some of the inputs to the time series generators are the number of inputs variable which is the number of data points needed to capture the time pattern of the data. This setting needs domain knowledge to properly set in a real world problem. For this demo, 24 months was selected to keep things simple. The other key input if the number of units that represents a single unit of time in the source data. Since this is a series based on months, the batch size selected is 1. If there were multiple observations for a single month, and the goal was a monthly forecast, then this value would need to be adjusted accordingly. 

The last part of the time series generator set up is to test the generator output. The training generator should generate a tensor equal to the number of inputs variable from above. The validation generator should generate one less than the training generator. 

```
###~~~
#use a time series generator to prep & stage the data for modelling
###~~~

#length of data points needed for the batches to properly capture the time series patterns
n_inputs = 24

#the number of data points to equal a single time unit in the source data
batchsize = 1

#determine generator inputs
train_generator = TimeseriesGenerator(data = scaled_train,
                                      targets = scaled_train,
                                      length = n_inputs,
                                      batch_size = batchsize,
                                      stride = 1,
                                      sampling_rate = 1,
                                      shuffle = False)

X_train, y_train = train_generator[0] #test the generator to see if it works
print(X_train, y_train) #print out the results
len(X_train[0]) #should equal the length value above

#resize the length this needs to be 1 less than the test set
n_inputs_val = n_inputs - 1

#create validation set generator
validation_generator = TimeseriesGenerator(data = scaled_test,
                                           targets = scaled_test,
                                           length = n_inputs_val,
                                           batch_size = batchsize,
                                           stride = 1,
                                           sampling_rate = 1,
                                           shuffle = False)

X_val, y_val = validation_generator[0] #test the generator to see if it works
print(X_val, y_val) #print out the results
len(X_val[0]) #should equal the length value above
```

This screen shot shows 24 at the bottom which is the correct value for the training generator configuration from above.

<img width="269" alt="image" src="https://github.com/garth-c/python_forecasting/assets/138831938/ebce032c-2f73-4ad5-831f-80f9093b7237">


This screen shot shows 23 at the bottom which is the correct value for the validation generator configuration from above.

<img width="271" alt="image" src="https://github.com/garth-c/python_forecasting/assets/138831938/760d8cfe-41ba-4bcd-8bf1-85332d1238fa">

------------------------------------------------------------------------------------------------------------------------------------------------------

# build the model

The next step is to configure a bidirectional model. Multiple decisions have to be made around the proper configuration of the model. Since this is a univariate model (only one predictor input), then the number of features is one. But if this was a multivariate model then this number would flex with the count of predictor variables that are being fed into the model. Another key input to the model is then number of neurons to use. The larger the count the longer the training time which usually leads to an overfit model. 

Since this is a bidirectional model, the data is processed forward and then backward to learn the pattern by the neurons. Both directions are noted in the model layers shown below in the code. The last layer of this model is a dense layer with an output of 1 which is the end result of the processing. In addition, an optimizer needs to be configured and included in the model set up process. 

The next few steps are to compile the model and then double check the model configuration  with the summary command. The last few steps are to configure an early stopping criteria based on val loss and then to set up the maximum number of model iterations with an epoch variable. The the final step to fit the data to the model. This is accomplished using the time series generators set up above. 

Once the fit command is executed, the model will process the source data using all of the parameters set up above. 

```
###~~~
#build the model
###~~~

#model input features
n_features = 1 #univariate = only one input feature from train set - this is the # of predictors
n_neurons = 100 #these need to be related to set up dynamic object: more = longer train times

#instantiate the stacked model type
model_LTSM_bd = Sequential()

#define the model
#ltsm layer 1 - forward
model_LTSM_bd.add(Bidirectional(LSTM(n_neurons,
                                     activation = 'relu',
                                     return_sequences = True),
                                     input_shape= (n_inputs, n_features)))
#lstm layer 2 - backward
model_LTSM_bd.add(Bidirectional(LSTM(n_neurons)))
#final dense layer
model_LTSM_bd.add(Dense(1))

#config the optimizer ~ use adaptive learning rate
opt = tf.keras.optimizers.Adam(learning_rate = 0.001,
                               beta_1= 0.9,
                               beta_2 = 0.999,
                               #decay = 0.001,
                               epsilon = 1e-08)

#compile the model
model_LTSM_bd.compile(optimizer = opt,
                     loss = 'mse')

#confirm model config
model_LTSM_bd.summary()

#set the max number of model iterations
n_epoch = 100

#early stopping config ~ validation loss no change for n epochs
early_stop = EarlyStopping(monitor = 'val_loss',
                           patience = 10)

#fit model to data
model_LTSM_bd.fit(train_generator,
                  epochs = n_epoch,
                  validation_data = validation_generator,
                  callbacks = [early_stop],
                  verbose = 1,
                  shuffle = False,
                  workers = 1,
                  use_multiprocessing = False)
```

The model summary is shown below. Note that this is a large number of paramters being used and training time will take a while. 

<img width="314" alt="image" src="https://github.com/garth-c/python_forecasting/assets/138831938/f3ce3465-fe83-43bd-98cc-63618754347e">

While model is processing this data, the epoch processing is detailed below. 

<img width="425" alt="image" src="https://github.com/garth-c/python_forecasting/assets/138831938/df52b33e-31d2-43a1-aa28-064857289e59">

The early stopping criteria stopped the processing after 11 epochs


-----------------------------------------------------------------------------------------------------------------------------------------------------------

# evaluate the model losses

The next step is to evaluate the model losses. The model losses per epoch processed are fed back into the model and then it makes course corrections based on the optimizer to adjust the weights of the neurons for the next epoch. Over the course of the processing, the model 'learns' the weights to use to get the best outcome for the forecast.

```
#eval the losses
losses = pd.DataFrame(model_LTSM_bd.history.history)
losses.plot(figsize = (10, 8))
plt.show(block = True)
```

A plot of the losses is below. As can be seen from this plot, the loss metric significantly flattens out after 4 epoch and the validation loss metric is still climbing. This indicates the model may be overfitting or the model could use more tuning. 

![model_losses](https://github.com/garth-c/python_forecasting/assets/138831938/c214496e-217b-440c-9eae-d56c006fb1db)


----------------------------------------------------------------------------------------------------------------------------------------

# create a forecast training loop

The next step is to create a training loop process to forecast one time unit ahead (month, t + 1) and then feed that new value back into the model for the next forecast time unit ahead (month,  t + 2). This lop will process for the entire test set length which is 24 months in this case. 

```
###~~~
#create a training loop for the forecast
###~~~

#create a place-holder list
test_predictions = []

#what does the first eval into the future look like
#last batch in the data set
first_eval_batch = scaled_train[-n_inputs:]

#reshape the list to meet the 3D tensor requirements
current_batch = first_eval_batch.reshape((1, n_inputs, n_features))

#test prediction loop
for i in range(len(test)):
    current_pred = model_LTSM_bd.predict(current_batch)[0]
    test_predictions.append(current_pred)
    current_batch = np.append(current_batch[:,1:,:],
                              [[current_pred]],
                              axis = 1)

#look at the test predictions
test_predictions
type(test_predictions)

#put predictions into a data frame
test_predictions_df = pd.DataFrame(test_predictions,
                                   columns = ['predictions'])

#look at the true prediction values
true_predictions = scaler.inverse_transform(test_predictions)
print(true_predictions)
type(true_predictions)
true_predictions_df = pd.DataFrame(true_predictions,
                                   columns = ['true_preds'])

#add true predictions to the raw test set
test['LSTM Predictions'] = true_predictions
print(test)
```

The output of the training loop from within the console is shown below. 

<img width="249" alt="image" src="https://github.com/garth-c/python_forecasting/assets/138831938/9f226422-c388-45f0-94b4-e51a9fac0c91">

After running the training loop the forecast output is put into a new Pandas data frame. These are all scaled values so to get the output converted back into the proper units for the source data and the objective of the project (head counts), the inverse transform function will be applied to the training output loop. The output of the inverse scaling is shown below. These values contain decimals and since this is a headcount project, the forecast values will ultimately need to be rounded to an integer as we can't have partial heads in a headcount forecast. 

<img width="85" alt="image" src="https://github.com/garth-c/python_forecasting/assets/138831938/db7389a2-d55d-43e7-a11a-d78677eb4a8e">

The end result of this transformation back into head counts and the comparison to the actual heads for the back testing is shown below. 

<img width="192" alt="image" src="https://github.com/garth-c/python_forecasting/assets/138831938/69c29682-4e18-4a06-9033-67c278730f1f">


---------------------------------------------------------------------------------------------------------------------------------------------------

# backtest the predictions 

Since this was random source data, the pattern value was negligible. So looking at the model output compared the actual values for the same time period is the backtesting. Looking at the random pattern of the actual data (blue line) and comparing it to the model output (red line), it is clear that the model could use more training to better capture the source data pattern. This could be adding more neurons, relaxing the early stopping parameters, using a different accuracy metric, using different otpimizer inputs, more epochs, etc. There are other transformations that could also be applied in addition to adding other predictor inputs which would make this a multivariate model. Either way, there are a lot of levers and knobs that could be used to tune the model and get better results. 

Howoever, overall the model captured the directional features of the random data set with much less pronounced slope changes and less pronounced spiking in the data. Thus, the model was pretty effective at capturing the general theme of the data despite the source data being random. 

![backtesting](https://github.com/garth-c/python_forecasting/assets/138831938/a8c8181c-f56b-4ee5-ac46-1c558609d898)


```
###~~~
#backtesting and house-keeping
###~~~

#visual check of test predictions compared to actual test values
test.plot(figsize = (12, 8))
plt.show(block = True)

#numerical eval rmse - closer to zero the better
np.sqrt((mean_squared_error(test['head_count'],
                            test['LSTM Predictions'])))

#save the model
from keras.models import load_model
model_LTSM_bd.save('model_LTSM_bd.h5')
```

The last things to do are to calcualte the mean squared error (MSE) metric to use for comparing against other models and the finally saving the model to reconstitute at a later date if needed. The MSE metric is 399.17 and it is shown below. 

<img width="269" alt="image" src="https://github.com/garth-c/python_forecasting/assets/138831938/81e9b083-328f-4239-9496-f56b9d340ac1">


The last bit of this code saves the model locally using the .h5 file format but there are other methods such as pickle that could also be used. 

Thanks for reading!

----------------------------------------------------------------------------------------------------------------------------------------------------------------------



