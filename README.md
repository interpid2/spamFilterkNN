Spam filter of sms messages using k nearest neighbours as a classifier.

SMS messages are provided in the english_big.txt file.

kNN classifier uses following statistics for input:
- Number of characters per SMS (message can be made out of multiple SMS of 160 characters long)
- percentage of non-AlphaNumeric symbols in comparison to number of characters typed
- percentage of numbers in comparison to number of characters typed (123 is counted as 3 numbers)
- are there unknown characters in the message (1 - yes, 0 - no)
- number of SMS sent


Script is made in Spyder IDE using Anaconda Distribution and Python 2.7. Link to Anaconda: www.anaconda.com/download

Before running the script, type nltk.download() in IPython terminal which will bring the download window. Download punkt(tokenizer model).
