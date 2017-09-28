Spam filter of sms messages using k nearest neighbours as a classifier.
SMS messages are provided in the english_big.txt file.
kNN classifier uses following statistics for input: Number of characters per sms (SMS can be made out of multiple messages of 160 characters long), percentage of non-AlphaNumber symbols in comparison to number of charachters typed, percentage of numbers in comparison to to number of characters typed (123 is counted as 3 numbers), are there unknow characters in SMS (1 - yes, 0 - no) and number of messages sent.


Script is made in Spyder IDE using Anaconda Distribution and Python 2.7. Link to Anaconda: www.anaconda.com/download

Before running the script, type nltk.download() in IPython terminal which will bring the download window. Download punkt(tokenizer model).
