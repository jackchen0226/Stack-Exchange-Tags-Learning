import pandas as pd
import numpy as np
from all_tags import all_tags
from bs4 import BeautifulSoup
import nltk

Test = pd.read_csv('data/robotics.csv')
# Find collocations for title and content
t_tokens = []
c_tokens = []
for i in range(Test.shape[0]):
    t_tokens.append(nltk.word_tokenize(BeautifulSoup(Test['title'].iloc[i]).get_text()))
    c_tokens.append(nltk.word_tokenize(BeautifulSoup(Test['content'].iloc[i]).get_text())) t_text = ntlk.Text(t_tokens)

'''
for i in range(Test.shape[0]):

    # To remove newline characters and to use nltk's Text type
    text = nltk.Text(tokens)
'''

