# sentiment-rnn

## Problem
Predicte the sentiment analysis of movie reviews. Here we'll use a dataset of movie reviews, accompanied by sentiment labels: positive or negative.

## Why to use LSTM: 
To care more about the context.

<img src="https://github.com/MuhammadAlBarham/sentiment-rnn/blob/main/assets/reviews_ex.png" width=50% height=50%>


## Data pre-processing¶

The first step when building a neural network model is getting your data into the proper form to feed into the network. Since we're using embedding layers, we'll need to encode each word with an integer. We'll also want to clean it up a bit.

You can see an example of the reviews data above. Here are the processing steps, we'll want to take:

We'll want to get rid of periods and extraneous punctuation.
Also, you might notice that the reviews are delimited with newline characters \n. To deal with those, I'm going to split the text into each review using \n as the delimiter.
Then I can combined all the reviews back together into one big string.
First, let's remove all punctuation. Then get all the text without the newlines and split it into individual words.

### Encoding the words & the labels

The embedding lookup requires that we pass in integers to our network. The easiest way to do this is to create dictionaries that map the words in the vocabulary to integers. Then we can convert each of our reviews into integers so they can be passed into the network.

### Removing Outliers

As an additional pre-processing step, we want to make sure that our reviews are in good shape for standard processing. That is, our network will expect a standard input text size, and so, we'll want to shape our reviews into a specific length. We'll approach this task in two main steps:

Getting rid of extremely long or short reviews; the outliers
Padding/truncating the remaining data so that we have reviews of the same length.
Before we pad our review text, we should check for reviews of extremely short or long lengths; outliers that may mess with our training.

## Padding sequences

To deal with both short and very long reviews, we'll pad or truncate all our reviews to a specific length. For reviews shorter than some seq_length, we'll pad with 0s. For reviews longer than seq_length, we can truncate them to the first seq_length words. A good seq_length, in this case, is 200.

## Training, Validation, Test

* Training: 80% of the data
* Validation: 10% of the data
* Test: 10% of the data

## DataLoaders and Batching

fter creating training, test, and validation data, we can create DataLoaders for this data by following two steps:

Create a known format for accessing our data, using TensorDataset which takes in an input set of data and a target set of data with the same first dimension, and creates a dataset.
Create DataLoaders and batch our training, validation, and test Tensor datasets.
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
train_loader = DataLoader(train_data, batch_size=batch_size)
This is an alternative to creating a generator function for batching our data into full batches.

## Sentiment Network with PyTorch: Architecture
![](https://github.com/MuhammadAlBarham/sentiment-rnn/blob/main/assets/network_diagram.png)

## Training

* We'll also be using a new kind of cross entropy loss, which is designed to work with a single Sigmoid output. BCELoss, or Binary Cross Entropy Loss, applies cross entropy loss to a single value between 0 and 1.
* The results are very good.
* Number of epochs: 4

## Testing

There are a few ways to test your network.

Test data performance: First, we'll see how our trained model performs on all of our defined test_data, above. We'll calculate the average loss and accuracy over the test data.
Inference on user-generated data: Second, we'll see if we can input just one example review at a time (without a label), and see what the trained model predicts. Looking at new, user input data like this, and predicting an output label, is called inference.


## Download the data:
The data can be downloaded from here: [data](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/sentiment-rnn/data).


 ## Referance: 
  
  This work depends on this project, [here](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/sentiment-rnn)
