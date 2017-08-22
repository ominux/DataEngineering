#------------------------------------------------------------------------------------------------
# Preprocessing
#------------------------------------------------------------------------------------------------
# Save everything in a folder called downloadedDataset
mkdir downloadedDataset
# Copy all the useful scripts
cp processWikipedia.pl ./downloadedDataset/.
# Move to that directory
cd downloadedDataset
#------------------------------------------------------------------------------------------------
# CIFAR 10 data
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvzf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz
# Result: Folder
# cifar-10-batches-py
#------------------------
# CIFAR 100 data
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xvzf cifar-100-python.tar.gz
rm cifar-100-python.tar.gz
# Result: Folder
# cifar-100-python
#------------------------------------------------------------------------------------------------
# text8, A subset of wikipedia
http://mattmahoney.net/dc/text8.zip
unzip text8.zip
rm text8.zip
#------------------------------------------------------------------------------------------------
# A subset of wikipedia, first billion characters from Wikipedia, 300MB
wget http://mattmahoney.net/dc/enwik9.zip 
# Unzip to get the xml file dumped by Wikipedia 
unzip enwik9.zip
rm enwik9.zip
# Parse the xml files to obtain the words. A perl script by http://mattmahoney.net/dc/textdata.html
perl processWikipedia.pl enwik9 > enwik9Data.txt
rm enwik9
# Result:
# enwik9Data.txt now contains a bunch of words that you can parse through to create a dictionary.
# ./enwik9Data.txt
#------------------------------------------------------------------------------------------------
# Gets a billion words that is already cleaned, 1.7GB
wget http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
tar -xvzf 1-billion-word-language-modeling-benchmark-r13output.tar.gz
rm 1-billion-word-language-modeling-benchmark-r13output.tar.gz
# Result: 
# The folder below now contains files, each contains many lines of sentences.
# ./1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled
#------------------------------------------------------------------------------------------------
# UMBC WebBase corpus of 3B English words, 13.2GB
# http://ebiquity.umbc.edu/blogger/2013/05/01/umbc-webbase-corpus-of-3b-english-words/
# High quality english paragraph consisting of 3 Billion words. 
wget http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
# unzip it, however, note libraries like gensim can work directly with .bz2 files
bzip2 -dk enwiki-latest-pages-articles.xml.bz2
# Result:
perl processWikipedia.pl enwiki-latest-pages-articles.xml > enwiki-latest-pages-articlesData.txt
# rm enwiki-latest-pages-articles.xml.bz2 # Do not remove since can use bz2 directly with gensim
#------------------------------------------------------------------------------------------------
