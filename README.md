# DataEngineering
My personal repository for working with data. 

## Scraping - Collection and Storing - Scraping data from the web.
    - Downloading into compressed
    - Uncompress files

- Save into disk with wanted format
    - Load the files into memory in a format wanted (numImage, width, height)
    - Store as pickle files to limit amount of in memory used.

## Cleaning
- Load from Disk and display to ensure data is fine
    - Load pickle files
    - Display the images

- Divide into train, val, test sets
    - After dividing, randomize the sequence in each set itself.
    - Need to randomize when running sequential algorithms on each set.

- Display divided data with their labels (for supervised learning)
    - Look for overlaps programmatically  cause the dataset may contain data that is close to identical.
    - If overlaps are significant, remove them. 

## Statistics
- Understanding the mean, variance, size of data. 
  - This allows further analysis
    - Which algorithm would be suitable
    - The metric for the scoring function and loss function
    - Establish a baseline score by using the simplest model,
      this shows if task is too easy or errors in data
      if baseline doesn't meet expectation.
- Check if dataset is skewed 
   - e.g. Too many examples for a class in a Classification task
   - If it is skewed,
       use F-score instead of Accuracy = (numCorrect/numPredict)

precision = (truePositive)/(truePositive + falsePositive)
recall = (truePositive)/(truePositive + falseNegative)
F-score = 2*(precision*recall)/(precision+recall)

## Algorithms
- Code for the various machine learning algorithms

## Visualization
- Display output from algorithm and accuracy.
- Visualize data and errors.
- Analyze how hyperparameters affect performance
- Hyperparamters:
  - Size of training data
  - Number of weights, hidden layers
- D3.js

## Library
- Tutorials on how to work with a particular library 
###  Tensorflow


## Installation Instructions
```bash
bash install.sh
```
