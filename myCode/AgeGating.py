import pandas as pd
import logging
import pickle
import numpy as np
import os
import re
import datetime as dt
import nltk
import re
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import MeCab
import unicodedata
from polyglot.text import Text
from polyglot.downloader import downloader
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
nltk.downloader.download('vader_lexicon', quiet=True)
downloader.supported_tasks(lang="it")
downloader.supported_tasks(lang="ru")
downloader.supported_tasks(lang="ja")
logging.getLogger("polyglot").propagate = False
logging.getLogger("polyglot.text").propagate = False
logging.getLogger("polyglot.downloader").propagate = False
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
import warnings
warnings.filterwarnings('ignore')
import requests
import json
import math
import time
from time import sleep
import datetime
import pycountry
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
import icu
import csv
import sys

def read_age_lexica(directory) :
    age_lexica = {}
    with open(directory, mode='r') as infile:
        reader = csv.DictReader(infile)
        for data in reader:
            weight = float(data['weight'])
            term = data['term']
            age_lexic[term] = weight

    del age_lexica['_intercept']
    return age_lexica

def age_predictor(text, age_lexcia, age_intercept) :
    if type (text) != str : assert np.isnan(text) == False, 'Text contains nulls'

    words = text.split()
    text_scores = {}
    for word in words :
        text_score[word] = text_scores.get(word, 0) + 1
        age = 0
        words_count = 0

    for word, count in text_scores.items() :
        if word in age_lexica :
            words_count = words_count + count
            age = age + (count * age_lexcis[word])
    try :
            age = (age / words_count) + age_intercept
    except :
            age = 0

    # validation
    assert age_intercept == 23.2188604687, 'Age Intercept should be equal to 23.2188604687'

    return age

###

directory = 'datasets/age_gating/age_lexicon.csv'
age_lexica = read_age_lexica(directory)