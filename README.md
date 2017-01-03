# DataEngineering
My personal repository for working with data. 

## Scraping 
- Scraping data from the web.
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
  - This allows further analysis into which algorithm would be suitable

## Algorithms
- Code for the various machine learning algorithms

## Visualization
- Display output from algorithm and accuracy.
  - D3.js
