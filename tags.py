import pandas as pd
import numpy as np
from all_tags import all_tags
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import bprofile
import csv

def main():
    Test = pd.read_csv('data/robotics.csv')
    all_tags = " ".join(Test['tags'].tolist())
    all_tags = list(set(nltk.word_tokenize(all_tags)))
    print(all_tags)
    stop = list(set(stopwords.words('english')))
    # Find collocations for title and contentTags
    q_tokens = []
    def inner():
        for i in range(Test.shape[0]):
            # Split off the lists for now; will be needed to collocations later. Return one list for wordInContent()
            t_tokens = nltk.word_tokenize(BeautifulSoup(Test['title'].iloc[i], "html5lib").get_text())
            c_tokens = nltk.word_tokenize(BeautifulSoup(Test['content'].iloc[i], "html5lib").get_text())
            q_tokens.append(t_tokens)
            #q_tokens.append(c_tokens)
            for x in stop:
                while x in q_tokens:
                    q_tokens.remove(x)
                    print('test')

    def wordInContent(text, tags):
        """Finds out if the word of a tag is within the texts and writes it to a csv file without any tags"""
        stackTags = []
        for x in range(len(text)):
            contentTags = []
            for i in range(len(tags)):
                for j in tags[i]:
                    if j in text:
                        if all_tags[i] not in contentTags:
                            contentTags.append(all_tags[i])
                            break
            stackTags.append(" ".join(contentTags))

        write = pd.read_csv("data/roboticsTagless.csv")
        write['tags'] = pd.Series(stackTags, index=write.index)
        print(write)

        # Tokenize words from tags
    def tokenizeTags(tags):
        tokens = []
        for i in tags:
            tokens.append(i.split("-"))
        return tokens

    print('ready')
    inner()
    print('tokensdone')
    wordInContent(q_tokens, tokenizeTags(all_tags))

if __name__ == '__main__':
    with bprofile.BProfile("profile2.png"):
        main()
