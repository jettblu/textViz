from nltk import *
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from math import log

import re
import pickle
import pandas as pd
import numpy as np
from collections import OrderedDict
import copy
from statistics import stdev
import string


def getTokens(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return tokens


def processTextChain(texts):
    processedTexts = []
    for text in texts:
        processedText = processText(text)
        processedTexts.append(processedText)
    return processedTexts


# processing of word informed by https://github.com/jaredks/tweetokenize/blob/master/tweetokenize/tokenizer.py
def processWord(word):
    rePunc = r'[{}]'.format(string.punctuation)
    reLink = r'https?:\/\/.*\/\w*'
    # make word lowercase
    word = word.lower()
    word = re.sub(rePunc, '', word)
    # remove links
    word = re.sub(reLink, '', word)

    return word


def processText(text):
    processedText = []
    ps = PorterStemmer()
    lem = WordNetLemmatizer()
    stopWords = set(stopwords.words('english') + list(string.punctuation))
    for word in text:
        word = processWord(word)
        processedText.append(lem.lemmatize(word))
    processedText = [word for word in processedText if word not in stopWords]
    return processedText


# save tokenized text samples as .npy
def saveTextSamples():
    data_file = 'main/textAnalysis/language/text_emotion.csv'
    data = pd.read_csv(data_file)
    del data['tweet_id']
    del data['author']

    positive = []
    negative = []

    for i, text in enumerate(data['content']):
        tokens = getTokens(text)
        sentiment = data['sentiment'][i]
        '''
        sentiments included in data: 'fun', 'neutral', 'love', 'boredom', 
        'relief', 'worry', 'sadness', 'empty', 'surprise', 'happiness', 
        'anger', 'hate', 'enthusiasm'
        '''
        # separate texts into positive/ negative categories
        if sentiment in ['fun', 'love', 'relief', 'surprise', 'happiness', 'enthusiasm']:
            positive.append(processText(tokens))
        if sentiment in ['boredom', 'worry', 'sadness', 'empty', 'anger', 'hate']:
            negative.append(processText(tokens))

    positiveSamples = np.array(positive)
    negativeSamples = np.array(negative)

    np.save('main/textAnalysis/language/negative samples.npy', negativeSamples)
    np.save('main/textAnalysis/language/positive samples.npy', positiveSamples)


def createUnigramLexDict():
    f = open("main/textAnalysis/language/lexicon/unigram lexicon.txt", "r")
    lexicon = dict()
    for line in f:
        # try/ except handles blank line(s) at end of unigram lexicon file
        try:
            wordAndScore = f.readline().split(" ")

            word, score = wordAndScore[0], float(wordAndScore[1].strip('\n'))

            # adds word/score pair to word:score dictionary
            lexicon[word] = score
        except:
            print("End of file reached.")
    f.close()
    with open('main/textAnalysis/language/lexicon/unigramLexDict.pickle', 'wb') as lexFile:
        pickle.dump(lexicon, lexFile)


def loadUnigramLexicon():
    with open('main/textAnalysis/language/lexicon/unigramLexDict.pickle', 'rb') as lexFile:
        lexicon = pickle.load(lexFile)
    return lexicon


def loadPositivesamples():
    return np.load('main/textAnalysis/language/positive samples.npy', allow_pickle=True)


def loadNegativeSamples():
    return np.load('main/textAnalysis/language/negative samples.npy', allow_pickle=True)


def getSynonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return synonyms


'''TF-IDF implementation guided by https://en.wikipedia.org/wiki/Tf%E2%80%93idf'''


# receives all samples as input and returns a dictionary
# containing the document frequency for each word
def createIDf(samples):
    docFreq = OrderedDict()
    numWords = 0
    for sample in samples:
        for word in sample:
            numWords += 1
            # counts number of times a given word appears in doc
            docFreq[word] = docFreq.get(word, 0) + 1

    for word in docFreq:
        docFreq[word] = log(numWords/docFreq[word])

    # remove low frequency words
    keys = [key for key in docFreq if (docFreq[key] > 10) or (docFreq[key]<5)]
    for key in keys:
        del docFreq[key]

    with open('main/textAnalysis/language/inverseDocFreq.pickle', 'wb') as lexFile:
        pickle.dump(docFreq, lexFile)


def loadIDF():
    try:
        with open('main/textAnalysis/language/inverseDocFreq.pickle', 'rb') as lexFile:
            docFreq = pickle.load(lexFile)
            return docFreq
    except:
        print("Issue loading IDF file. Check to see if file exists.")


createIDf(np.concatenate((loadPositivesamples(), loadPositivesamples()), axis=0))
docFreqMaster = loadIDF()


# creates vector from tokenized sample
def tfidfVectorizer(sample):
    termFrequencies = OrderedDict()
    numWords = len(sample)
    docFreq = copy.copy(docFreqMaster)
    for word in sample:
        termFrequencies[word] = termFrequencies.get(word, 0) + 1
    for word in termFrequencies:
        termFrequencies[word] = termFrequencies[word]/numWords
    for term in docFreq:
        if term in termFrequencies:
            docFreq[term] = termFrequencies[term] * docFreq[term]
        else:
            docFreq[term] = 0
    # returns a sparse tfidf vector
    return list(docFreq.values())


# receives seconds as input
# returns hours, minutes, seconds
def convertSeconds(seconds):
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)

    min = int(min)
    sec = int(sec)
    hour = int(hour)

    if hour == 0:
        if min > 1 and sec > 1:
            return f'{min} minutes {sec} seconds'
        if min == 1 and sec > 1:
            return f'{min} minute {sec} seconds'
        if min > 1 and sec == 1:
            return f'{min} minutes {sec} second'
    elif min == 0:
        if sec > 1:
            return f'{sec} seconds'
        else:
            return f'{sec} second'
    else:
        if hour > 1 and min > 1 and sec > 1:
            return f'{hour} hours {min} minutes {sec} seconds'
        if hour == 1:
            return f'{hour} hour {min} minutes {sec} seconds'


# assumes input times are ms
# returns average number of seconds in list
def averageTime(timesList):
    return (sum(timesList)/len(timesList))/1000


# returns standard deviation of ms list in seconds
def stdDevTime(timesList):
    return stdev(timesList)/1000


# returns list of contacts arranged by text frequency
def sortContactFrequency(contactDict):
    return sorted(contactDict, key=lambda name: contactDict[name].textCount, reverse=True)


# returns list of contacts arranged alphabetically
def sortConatctsAlphabetically(contactsDict):
    return sorted(contactsDict.keys(), key=lambda x: x.lower())


def allContactsSummmary(contactsDict):
    incomingTextsCount = 0
    outgoingTextCount = 0
    allTexts = 0
    for contactName in contactsDict:
        contact = contactsDict[contactName]
        incoming = len(contact.incoming.texts)
        outgoing = len(contact.outgoing.texts)
        incomingTextsCount += incoming
        outgoingTextCount += outgoing
        allTexts += incoming + outgoing
    return incomingTextsCount, outgoingTextCount, allTexts
