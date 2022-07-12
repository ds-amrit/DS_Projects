# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 17:31:44 2022

@author: amrit
"""

import re
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string

import gc
import os

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from wordcloud import WordCloud,STOPWORDS, ImageColorGenerator

from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel

import torch
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast

from googletrans import Translator

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.getcwd()

csv_collection = []
for dirname, _, filenames in os.walk('E:\\TRENT\\SEM 4 Research Project\\new'):
    for filename in filenames:
        fullpath= os.path.join(dirname, filename)
        csv_collection.append(fullpath)

df = pd.read_csv(csv_collection.pop(),compression = 'gzip', index_col=0)
for data in csv_collection:
    try:
        tmp = pd.read_csv(data, compression = 'gzip', index_col=0)
    except:
            tmp = pd.read_csv(data, index_col = 0)
            df = pd.concat([df, tmp], axis=0)

#df.columns
df.shape


# Filtering relevant columns
newdf = df[['location','language', 'retweetcount','text','hashtags']]

# Checking the different unique languages present in the dataset
newdf.language.value_counts()

# Filtering English Language Tweets
eng_df = newdf.loc[newdf.language == "en","text"]

# Filtering Russian Language Tweets
ru_df = newdf.loc[newdf.language == "ru","text"]

#nltk.download('stopwords')

# Loading Stemmer model for english language
stemmer = nltk.SnowballStemmer("english")
# Creating a set of stopwords in english
stopword = set(stopwords.words('english'))

# Defing a function to clean the text
def cleantext(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text



# Cleaning the twerts in the english langauge
eng_clean = eng_df.apply(cleantext)

# Defing a function to Create a Wordcloud
def wc(tweets):
    text = " ".join(i for i in tweets)
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(stopwords=stopwords).generate(text)
    plt.figure(figsize = (15,10))
    plt.imshow(wordcloud,interpolation='bilinear')
    plt.axis('off')
    plt.show()
    
# Creating a wordcloud for english tweets
wc(eng_clean["text"])

# Creating a model to analyze sentiment intensity of the tweets using VADER lexicon
sentiments = SentimentIntensityAnalyzer()


eng_clean = pd.DataFrame(eng_clean)

# Calculating the sentiment intensity for english tweets
eng_clean["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in eng_clean["text"]]
eng_clean["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in eng_clean["text"]]
eng_clean["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in eng_clean["text"]]

# Creating a corpus of positive tweets in english language
positive = " ".join([i for i in eng_clean["text"][eng_clean["Positive"] > eng_clean["Negative"]]])

# Generating a WordCloud for positive tweets
pwordcloud = WordCloud(stopwords =stopword,background_color="white").generate(positive)
plt.figure(figsize = (15,10))
plt.imshow(pwordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()

# Creating a corpus of negative tweets in english language
negative = " ".join([i for i in eng_clean["text"][eng_clean["Positive"] < eng_clean["Negative"]]])

# Generating a WordCloud for Negative tweets
nwordcloud = WordCloud(stopwords =stopword,background_color="white").generate(negative)
plt.figure(figsize = (15,10))
plt.imshow(nwordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()

### SENTIMENT ANALYSIS OF RUSSIAN TEXT 

## METHOD 1: By translating text to english and calculating sentiment

# Creating a model to call google translator API for translating text 
translator = Translator()
 
# Translating Tweets in russian language to english language
ru_eng = []
for i in ru_df:
    ru_eng.append(translator.translate(i).text)

# Filtering unique tweets as the dataset contains retweets as well.
ru_eng_u = list(dict.fromkeys(ru_eng))
ru_eng_clean = []
for i in ru_eng_u:
    ru_eng_clean.append(cleantext(i))
    
# Creating a dataframe of the translated tweets 
ru_eng_clean = pd.DataFrame(ru_eng_clean,columns=["text"])

# Saving to a file to avoid re-translation of text
ru_eng_clean.to_csv("E:\\TRENT\\SEM 4 Research Project\\ru_2_en.csv",sep=",",index=False)

ru_eng_df = pd.read_csv("E:\\TRENT\\SEM 4 Research Project\\ru_2_en.csv")

# Calculating the sentiment intensity for english translated tweets
ru_eng_clean["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in ru_eng_clean["text"]]
ru_eng_clean["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in ru_eng_clean["text"]]
ru_eng_clean["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in ru_eng_clean["text"]]
ru_eng_clean["compound"] = [sentiments.polarity_scores(i)["compound"] for i in ru_eng_clean["text"]]


# Defing a function to classify tweets as postive,negative or neutral.   
def get_analysis(score):
    if score >= 0.05:
        return 'Positive'
    elif score > -0.05 and score < 0.05:
        return 'Neutral'
    elif score <= -0.05:
        return 'Negative'
    
# Creating a new column which will contain the final sentiment of the tweets
ru_eng_clean['Analysis'] = ru_eng_clean["compound"].apply(get_analysis)

## show the distribution of sentiment with count
ru_eng_clean['Analysis'].value_counts()

## plot visulatisation of count
plt.title('Sentiment Analysis of Russian text translated to English')
plt.xlabel('Sentiments') 
plt.ylabel('Counts')
ru_eng_clean['Analysis'].value_counts(ascending=True).plot(kind='bar')
plt.show()


## METHOD 2: Using the dostoevsky library 

# Creating a tokenizer for preprocessing russian text
tokenizer = RegexTokenizer()

# Creating a model to analyze sentiment of russian text
model = FastTextSocialNetworkModel(tokenizer=tokenizer)

# Creating a unique list of russian tweets
ru_df_u = list(dict.fromkeys(ru_df))

# Creating a DataFrame of unique russian tweets
r_df = pd.DataFrame(ru_df_u,columns=["text"])

# Calculating the sentiment intensity of Russian tweets
results = model.predict(ru_df_u,k=2)


# Creating new columns based on the result of sentiment intensity
r_df["Neutral"] = [sentiment.get("neutral") for sentiment in results]
r_df["Negative"] = [sentiment.get("negative") for sentiment in results]
r_df["Positive"] = [sentiment.get("positive") for sentiment in results]
r_df["skip"] = [sentiment.get("skip") for sentiment in results]
r_df["speech"] = [sentiment.get("speech") for sentiment in results]

# Checking for null values
r_df.isnull().sum()

# Replacing all the null values in the result with 0 as it denotes 0 confidence.
r_df.replace(np.nan,0,inplace = True)

# Classifying tweets at 10% confidence level
r_df["Analysis"] = ""
for i in range(r_df.shape[0]):
    if r_df["Positive"][i] > 0.1:
        r_df["Analysis"][i] = "Positive"
    elif r_df["Negative"][i] > 0.1:
        r_df["Analysis"][i] = "Negative"
    else:
        r_df["Analysis"][i] = "Neutral"


r_df.Analysis.value_counts()

plt.title('Sentiment Analysis of Russian Text using Dostoevsky')
plt.xlabel('Sentiments') 
plt.ylabel('Counts')
r_df.Analysis.value_counts(ascending=True).plot(kind='bar')

## METHOD 3: Analyzing the sentiment intensity of russian tweets using the MODIFIED VADER 
#                                        for russian texts.

r_df_scores = pd.DataFrame(r_df.loc[:,"text"])

r_df_scores["Positive"] = [analyzer.polarity_scores(i)["pos"] for i in ru_df_u]
r_df_scores["Negative"] = [analyzer.polarity_scores(i)["neg"] for i in ru_df_u]
r_df_scores["Neutral"] = [analyzer.polarity_scores(i)["neu"] for i in ru_df_u]
r_df_scores["compound"] = [analyzer.polarity_scores(i)["compound"] for i in ru_df_u]

r_df_scores['Analysis'] = r_df_scores["compound"].apply(get_analysis)

## show value counts
r_df_scores['Analysis'].value_counts()
## plot visulatisation of count
plt.title('Sentiment Analysis of Russian text VADER Russian corpus')
plt.xlabel('Sentiments') 
plt.ylabel('Counts')
r_df_scores['Analysis'].value_counts(ascending=True).plot(kind='bar')
plt.show()


## METHOD 4: Using RuBERT Model from Hugging Face Transformers library

ru_bertdf = pd.DataFrame(r_df.loc[:,"text"])

tokenizer = BertTokenizerFast.from_pretrained('blanchefort/rubert-base-cased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('blanchefort/rubert-base-cased-sentiment', return_dict=True)

@torch.no_grad()
def predict(text):
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted = torch.argmax(predicted, dim=1).numpy()
    return predicted


def score(results):
    rl = []   
    for i in results:
        if i == [0]:
            rl.append("Neutral")
        elif i == [1]:
            rl.append("Positive")
        elif i == [2]:
            rl.append("Negative")
    return rl   

res = ru_bertdf.text.apply(predict)

ru_bertdf["Analysis"] = score(res)
ru_bertdf.Analysis.value_counts().plot(kind = 'bar')
plt.title("Sentiment Analysis of Russian Text using RuBERT transformer model")
plt.show()


## END

## Refactored Source code for VADER sentiment lexicon for russian language
"""
If you use the VADER sentiment analysis tools, please cite:
Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for
Sentiment Analysis of Social Media Text. Eighth International Conference on
Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
"""
import os
import re
import math
import string
import codecs
import json
from itertools import product
from inspect import getsourcefile
from io import open
from googletrans import Translator

translator = Translator()

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ##Constants##

# (empirically derived mean sentiment intensity rating increase for booster words)
B_INCR = 0.293
B_DECR = -0.293

# (empirically derived mean sentiment intensity rating increase for using ALLCAPs to emphasize a word)
C_INCR = 0.733
N_SCALAR = -0.74

NEGATE = \
    ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
     "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
     "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
     "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
     "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
     "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
     "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
     "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]
    
NEGATE_ru = translator.translate(NEGATE,dest='ru')
NEGATE_ru = [i.text for i in NEGATE_ru]

# booster/dampener 'intensifiers' or 'degree adverbs'
# http://en.wiktionary.org/wiki/Category:English_degree_adverbs

BOOSTER_DICT = \
    {"absolutely": B_INCR, "amazingly": B_INCR, "awfully": B_INCR,
     "completely": B_INCR, "considerable": B_INCR, "considerably": B_INCR,
     "decidedly": B_INCR, "deeply": B_INCR, "effing": B_INCR, "enormous": B_INCR, "enormously": B_INCR,
     "entirely": B_INCR, "especially": B_INCR, "exceptional": B_INCR, "exceptionally": B_INCR,
     "extreme": B_INCR, "extremely": B_INCR,
     "fabulously": B_INCR, "flipping": B_INCR, "flippin": B_INCR, "frackin": B_INCR, "fracking": B_INCR,
     "fricking": B_INCR, "frickin": B_INCR, "frigging": B_INCR, "friggin": B_INCR, "fully": B_INCR,
     "fuckin": B_INCR, "fucking": B_INCR, "fuggin": B_INCR, "fugging": B_INCR,
     "greatly": B_INCR, "hella": B_INCR, "highly": B_INCR, "hugely": B_INCR,
     "incredible": B_INCR, "incredibly": B_INCR, "intensely": B_INCR,
     "major": B_INCR, "majorly": B_INCR, "more": B_INCR, "most": B_INCR, "particularly": B_INCR,
     "purely": B_INCR, "quite": B_INCR, "really": B_INCR, "remarkably": B_INCR,
     "so": B_INCR, "substantially": B_INCR,
     "thoroughly": B_INCR, "total": B_INCR, "totally": B_INCR, "tremendous": B_INCR, "tremendously": B_INCR,
     "uber": B_INCR, "unbelievably": B_INCR, "unusually": B_INCR, "utter": B_INCR, "utterly": B_INCR,
     "very": B_INCR,
     "almost": B_DECR, "barely": B_DECR, "hardly": B_DECR, "just enough": B_DECR,
     "kind of": B_DECR, "kinda": B_DECR, "kindof": B_DECR, "kind-of": B_DECR,
     "less": B_DECR, "little": B_DECR, "marginal": B_DECR, "marginally": B_DECR,
     "occasional": B_DECR, "occasionally": B_DECR, "partly": B_DECR,
     "scarce": B_DECR, "scarcely": B_DECR, "slight": B_DECR, "slightly": B_DECR, "somewhat": B_DECR,
     "sort of": B_DECR, "sorta": B_DECR, "sortof": B_DECR, "sort-of": B_DECR}
    
bd = list(BOOSTER_DICT.keys())
bd_ru = translator.translate(bd,dest='ru')
bd_ru = [i.text for i in bd_ru]

BOOSTER_DICT_ru ={}
for i in range(len(bd_ru)):
    BOOSTER_DICT_ru[bd_ru[i]] = BOOSTER_DICT.get(bd[i]) 
    

# check for sentiment laden idioms that do not contain lexicon words (future work, not yet implemented)
SENTIMENT_LADEN_IDIOMS = {"cut the mustard": 2, "hand to mouth": -2,
                          "back handed": -2, "blow smoke": -2, "blowing smoke": -2,
                          "upper hand": 1, "break a leg": 2,
                          "cooking with gas": 2, "in the black": 2, "in the red": -2,
                          "on the ball": 2, "under the weather": -2}

# check for special case idioms and phrases containing lexicon words
SPECIAL_CASES = {"the shit": 3, "the bomb": 3, "bad ass": 1.5, "badass": 1.5, "bus stop": 0.0,
                 "yeah right": -2, "kiss of death": -1.5, "to die for": 3,
                 "beating heart": 3.1, "broken heart": -2.9 }

sc =list(SPECIAL_CASES.keys())
sc_ru = translator.translate(sc,dest='ru')
sc_ru = [i.text for i in sc_ru]
SPECIAL_CASES_ru ={}
for i in range(len(sc_ru)):
    SPECIAL_CASES_ru[sc_ru[i]] = SPECIAL_CASES.get(sc[i])


# #Static methods# #

def negated(input_words, include_nt=True):
    """
    Determine if input contains negation words
    """
    input_words = [str(w).lower() for w in input_words]
    neg_words = []
    neg_words.extend(NEGATE_ru)
    for word in neg_words:
        if word in input_words:
            return True
    if include_nt:
        for word in input_words:
            if "нет" in word:
                return True
    '''if "least" in input_words:
        i = input_words.index("least")
        if i > 0 and input_words[i - 1] != "at":
            return True'''
    return False


def normalize(score, alpha=15):
    """
    Normalize the score to be between -1 and 1 using an alpha that
    approximates the max expected value
    """
    norm_score = score / math.sqrt((score * score) + alpha)
    if norm_score < -1.0:
        return -1.0
    elif norm_score > 1.0:
        return 1.0
    else:
        return norm_score


def allcap_differential(words):
    """
    Check whether just some words in the input are ALL CAPS
    :param list words: The words to inspect
    :returns: `True` if some but not all items in `words` are ALL CAPS
    """
    is_different = False
    allcap_words = 0
    for word in words:
        if word.isupper():
            allcap_words += 1
    cap_differential = len(words) - allcap_words
    if 0 < cap_differential < len(words):
        is_different = True
    return is_different


def scalar_inc_dec(word, valence, is_cap_diff):
    """
    Check if the preceding words increase, decrease, or negate/nullify the
    valence
    """
    scalar = 0.0
    word_lower = word.lower()
    if word_lower in BOOSTER_DICT_ru:
        scalar = BOOSTER_DICT_ru[word_lower]
        if valence < 0:
            scalar *= -1
        # check if booster/dampener word is in ALLCAPS (while others aren't)
        if word.isupper() and is_cap_diff:
            if valence > 0:
                scalar += C_INCR
            else:
                scalar -= C_INCR
    return scalar


class SentiText(object):
    """
    Identify sentiment-relevant string-level properties of input text.
    """

    def __init__(self, text):
        if not isinstance(text, str):
            text = str(text).encode('utf-8')
        self.text = text
        self.words_and_emoticons = self._words_and_emoticons()
        # doesn't separate words from\
        # adjacent punctuation (keeps emoticons & contractions)
        self.is_cap_diff = allcap_differential(self.words_and_emoticons)

    @staticmethod
    def _strip_punc_if_word(token):
        """
        Removes all trailing and leading punctuation
        If the resulting string has two or fewer characters,
        then it was likely an emoticon, so return original string
        (ie ":)" stripped would be "", so just return ":)"
        """
        stripped = token.strip(string.punctuation)
        if len(stripped) <= 2:
            return token
        return stripped

    def _words_and_emoticons(self):
        """
        Removes leading and trailing puncutation
        Leaves contractions and most emoticons
            Does not preserve punc-plus-letter emoticons (e.g. :D)
        """
        wes = self.text.split()
        stripped = list(map(self._strip_punc_if_word, wes))
        return stripped

class SentimentIntensityAnalyzer(object):
    """
    Give a sentiment intensity score to sentences.
    """

    def __init__(self, lexicon_file="vader_lexicon.txt", emoji_lexicon="emoji_utf8_lexicon.txt"):
        #_this_module_file_path_ = os.path.abspath(getsourcefile(lambda: 0))
        #lexicon_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), lexicon_file)
        lexicon_full_filepath = "E:\\TRENT\\SEM 4 Research Project\\vader_lexicon_ru.txt"
        with codecs.open(lexicon_full_filepath, encoding='utf-8') as f:
            self.lexicon_full_filepath = f.read()
        self.lexicon = self.make_lex_dict()

        #emoji_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), emoji_lexicon)
        emoji_full_filepath = "E:\\TRENT\\SEM 4 Research Project\\emoji_utf8_lexicon.txt"
        with codecs.open(emoji_full_filepath, encoding='utf-8') as f:
            self.emoji_full_filepath = f.read()
        self.emojis = self.make_emoji_dict()

    def make_lex_dict(self):
        """
        Convert lexicon file to a dictionary
        """
        lex_dict = {}
        for line in self.lexicon_full_filepath.rstrip('\n').split('\n'):
            if not line:
                continue
            (word, measure) = line.strip().split('\t')[0:2]
            lex_dict[word] = float(measure)
        return lex_dict

    def make_emoji_dict(self):
        """
        Convert emoji lexicon file to a dictionary
        """
        emoji_dict = {}
        for line in self.emoji_full_filepath.rstrip('\n').split('\n'):
            (emoji, description) = line.strip().split('\t')[0:2]
            emoji_dict[emoji] = description
        return emoji_dict

    def polarity_scores(self, text):
        """
        Return a float for sentiment strength based on the input text.
        Positive values are positive valence, negative value are negative
        valence.
        """
        # convert emojis to their textual descriptions
        text_no_emoji = ""
        prev_space = True
        for chr in text:
            if chr in self.emojis:
                # get the textual description
                description = self.emojis[chr]
                if not prev_space:
                    text_no_emoji += ' '
                text_no_emoji += description
                prev_space = False
            else:
                text_no_emoji += chr
                prev_space = chr == ' '
        text = text_no_emoji.strip()

        sentitext = SentiText(text)

        sentiments = []
        words_and_emoticons = sentitext.words_and_emoticons
        for i, item in enumerate(words_and_emoticons):
            valence = 0
            # check for vader_lexicon words that may be used as modifiers or negations
            if item.lower() in BOOSTER_DICT:
                sentiments.append(valence)
                continue
            if (i < len(words_and_emoticons) - 1 and item.lower() == "kind" and
                    words_and_emoticons[i + 1].lower() == "of"):
                sentiments.append(valence)
                continue

            sentiments = self.sentiment_valence(valence, sentitext, item, i, sentiments)

        sentiments = self._but_check(words_and_emoticons, sentiments)

        valence_dict = self.score_valence(sentiments, text)

        return valence_dict

    def sentiment_valence(self, valence, sentitext, item, i, sentiments):
        is_cap_diff = sentitext.is_cap_diff
        words_and_emoticons = sentitext.words_and_emoticons
        item_lowercase = item.lower()
        if item_lowercase in self.lexicon:
            # get the sentiment valence 
            valence = self.lexicon[item_lowercase]

            # check for "no" as negation for an adjacent lexicon item vs "no" as its own stand-alone lexicon item
            if item_lowercase == "no" and i != len(words_and_emoticons)-1 and words_and_emoticons[i + 1].lower() in self.lexicon:
                # don't use valence of "no" as a lexicon item. Instead set it's valence to 0.0 and negate the next item
                valence = 0.0
            if (i > 0 and words_and_emoticons[i - 1].lower() == "no") \
               or (i > 1 and words_and_emoticons[i - 2].lower() == "no") \
               or (i > 2 and words_and_emoticons[i - 3].lower() == "no" and words_and_emoticons[i - 1].lower() in ["or", "nor"] ):
                valence = self.lexicon[item_lowercase] * N_SCALAR

            # check if sentiment laden word is in ALL CAPS (while others aren't)
            if item.isupper() and is_cap_diff:
                if valence > 0:
                    valence += C_INCR
                else:
                    valence -= C_INCR

            for start_i in range(0, 3):
                # dampen the scalar modifier of preceding words and emoticons
                # (excluding the ones that immediately preceed the item) based
                # on their distance from the current item.
                if i > start_i and words_and_emoticons[i - (start_i + 1)].lower() not in self.lexicon:
                    s = scalar_inc_dec(words_and_emoticons[i - (start_i + 1)], valence, is_cap_diff)
                    if start_i == 1 and s != 0:
                        s = s * 0.95
                    if start_i == 2 and s != 0:
                        s = s * 0.9
                    valence = valence + s
                    valence = self._negation_check(valence, words_and_emoticons, start_i, i)
                    if start_i == 2:
                        valence = self._special_idioms_check(valence, words_and_emoticons, i)

            valence = self._least_check(valence, words_and_emoticons, i)
        sentiments.append(valence)
        return sentiments

    def _least_check(self, valence, words_and_emoticons, i):
        # check for negation case using "least"
        if i > 1 and words_and_emoticons[i - 1].lower() not in self.lexicon \
                and words_and_emoticons[i - 1].lower() == "least":
            if words_and_emoticons[i - 2].lower() != "at" and words_and_emoticons[i - 2].lower() != "very":
                valence = valence * N_SCALAR
        elif i > 0 and words_and_emoticons[i - 1].lower() not in self.lexicon \
                and words_and_emoticons[i - 1].lower() == "least":
            valence = valence * N_SCALAR
        return valence

    @staticmethod
    def _but_check(words_and_emoticons, sentiments):
        # check for modification in sentiment due to contrastive conjunction 'but'
        words_and_emoticons_lower = [str(w).lower() for w in words_and_emoticons]
        if 'but' in words_and_emoticons_lower:
            bi = words_and_emoticons_lower.index('but')
            for sentiment in sentiments:
                si = sentiments.index(sentiment)
                if si < bi:
                    sentiments.pop(si)
                    sentiments.insert(si, sentiment * 0.5)
                elif si > bi:
                    sentiments.pop(si)
                    sentiments.insert(si, sentiment * 1.5)
        return sentiments

    @staticmethod
    def _special_idioms_check(valence, words_and_emoticons, i):
        words_and_emoticons_lower = [str(w).lower() for w in words_and_emoticons]
        onezero = "{0} {1}".format(words_and_emoticons_lower[i - 1], words_and_emoticons_lower[i])

        twoonezero = "{0} {1} {2}".format(words_and_emoticons_lower[i - 2],
                                          words_and_emoticons_lower[i - 1], words_and_emoticons_lower[i])

        twoone = "{0} {1}".format(words_and_emoticons_lower[i - 2], words_and_emoticons_lower[i - 1])

        threetwoone = "{0} {1} {2}".format(words_and_emoticons_lower[i - 3],
                                           words_and_emoticons_lower[i - 2], words_and_emoticons_lower[i - 1])

        threetwo = "{0} {1}".format(words_and_emoticons_lower[i - 3], words_and_emoticons_lower[i - 2])

        sequences = [onezero, twoonezero, twoone, threetwoone, threetwo]

        for seq in sequences:
            if seq in SPECIAL_CASES:
                valence = SPECIAL_CASES[seq]
                break

        if len(words_and_emoticons_lower) - 1 > i:
            zeroone = "{0} {1}".format(words_and_emoticons_lower[i], words_and_emoticons_lower[i + 1])
            if zeroone in SPECIAL_CASES:
                valence = SPECIAL_CASES[zeroone]
        if len(words_and_emoticons_lower) - 1 > i + 1:
            zeroonetwo = "{0} {1} {2}".format(words_and_emoticons_lower[i], words_and_emoticons_lower[i + 1],
                                              words_and_emoticons_lower[i + 2])
            if zeroonetwo in SPECIAL_CASES:
                valence = SPECIAL_CASES[zeroonetwo]

        # check for booster/dampener bi-grams such as 'sort of' or 'kind of'
        n_grams = [threetwoone, threetwo, twoone]
        for n_gram in n_grams:
            if n_gram in BOOSTER_DICT:
                valence = valence + BOOSTER_DICT[n_gram]
        return valence

    @staticmethod
    def _sentiment_laden_idioms_check(valence, senti_text_lower):
        # Future Work
        # check for sentiment laden idioms that don't contain a lexicon word
        idioms_valences = []
        for idiom in SENTIMENT_LADEN_IDIOMS:
            if idiom in senti_text_lower:
                print(idiom, senti_text_lower)
                valence = SENTIMENT_LADEN_IDIOMS[idiom]
                idioms_valences.append(valence)
        if len(idioms_valences) > 0:
            valence = sum(idioms_valences) / float(len(idioms_valences))
        return valence

    @staticmethod
    def _negation_check(valence, words_and_emoticons, start_i, i):
        words_and_emoticons_lower = [str(w).lower() for w in words_and_emoticons]
        if start_i == 0:
            if negated([words_and_emoticons_lower[i - (start_i + 1)]]):  # 1 word preceding lexicon word (w/o stopwords)
                valence = valence * N_SCALAR
        if start_i == 1:
            if words_and_emoticons_lower[i - 2] == "never" and \
                    (words_and_emoticons_lower[i - 1] == "so" or
                     words_and_emoticons_lower[i - 1] == "this"):
                valence = valence * 1.25
            elif words_and_emoticons_lower[i - 2] == "without" and \
                    words_and_emoticons_lower[i - 1] == "doubt":
                valence = valence
            elif negated([words_and_emoticons_lower[i - (start_i + 1)]]):  # 2 words preceding the lexicon word position
                valence = valence * N_SCALAR
        if start_i == 2:
            if words_and_emoticons_lower[i - 3] == "never" and \
                    (words_and_emoticons_lower[i - 2] == "so" or words_and_emoticons_lower[i - 2] == "this") or \
                    (words_and_emoticons_lower[i - 1] == "so" or words_and_emoticons_lower[i - 1] == "this"):
                valence = valence * 1.25
            elif words_and_emoticons_lower[i - 3] == "without" and \
                    (words_and_emoticons_lower[i - 2] == "doubt" or words_and_emoticons_lower[i - 1] == "doubt"):
                valence = valence
            elif negated([words_and_emoticons_lower[i - (start_i + 1)]]):  # 3 words preceding the lexicon word position
                valence = valence * N_SCALAR
        return valence

    def _punctuation_emphasis(self, text):
        # add emphasis from exclamation points and question marks
        ep_amplifier = self._amplify_ep(text)
        qm_amplifier = self._amplify_qm(text)
        punct_emph_amplifier = ep_amplifier + qm_amplifier
        return punct_emph_amplifier

    @staticmethod
    def _amplify_ep(text):
        # check for added emphasis resulting from exclamation points (up to 4 of them)
        ep_count = text.count("!")
        if ep_count > 4:
            ep_count = 4
        # (empirically derived mean sentiment intensity rating increase for
        # exclamation points)
        ep_amplifier = ep_count * 0.292
        return ep_amplifier

    @staticmethod
    def _amplify_qm(text):
        # check for added emphasis resulting from question marks (2 or 3+)
        qm_count = text.count("?")
        qm_amplifier = 0
        if qm_count > 1:
            if qm_count <= 3:
                # (empirically derived mean sentiment intensity rating increase for
                # question marks)
                qm_amplifier = qm_count * 0.18
            else:
                qm_amplifier = 0.96
        return qm_amplifier

    @staticmethod
    def _sift_sentiment_scores(sentiments):
        # want separate positive versus negative sentiment scores
        pos_sum = 0.0
        neg_sum = 0.0
        neu_count = 0
        for sentiment_score in sentiments:
            if sentiment_score > 0:
                pos_sum += (float(sentiment_score) + 1)  # compensates for neutral words that are counted as 1
            if sentiment_score < 0:
                neg_sum += (float(sentiment_score) - 1)  # when used with math.fabs(), compensates for neutrals
            if sentiment_score == 0:
                neu_count += 1
        return pos_sum, neg_sum, neu_count

    def score_valence(self, sentiments, text):
        if sentiments:
            sum_s = float(sum(sentiments))
            # compute and add emphasis from punctuation in text
            punct_emph_amplifier = self._punctuation_emphasis(text)
            if sum_s > 0:
                sum_s += punct_emph_amplifier
            elif sum_s < 0:
                sum_s -= punct_emph_amplifier

            compound = normalize(sum_s)
            # discriminate between positive, negative and neutral sentiment scores
            pos_sum, neg_sum, neu_count = self._sift_sentiment_scores(sentiments)

            if pos_sum > math.fabs(neg_sum):
                pos_sum += punct_emph_amplifier
            elif pos_sum < math.fabs(neg_sum):
                neg_sum -= punct_emph_amplifier

            total = pos_sum + math.fabs(neg_sum) + neu_count
            pos = math.fabs(pos_sum / total)
            neg = math.fabs(neg_sum / total)
            neu = math.fabs(neu_count / total)

        else:
            compound = 0.0
            pos = 0.0
            neg = 0.0
            neu = 0.0

        sentiment_dict = \
            {"neg": round(neg, 3),
             "neu": round(neu, 3),
             "pos": round(pos, 3),
             "compound": round(compound, 4)}

        return sentiment_dict


analyzer = SentimentIntensityAnalyzer()