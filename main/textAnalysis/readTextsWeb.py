import xml.etree.ElementTree as ET
import string
import nltk
import joblib
from operator import itemgetter
import main.textAnalysis.utils as utils
import main.textAnalysis.messageAnalysis as messageAnalysis
import time as t


class Contact:
    contactsDict = {}

    def __init__(self, name, messageType, timeStamp):
        self.name = name
        self.messageTypes = []
        self.outgoing = None
        self.incoming = None
        self.textCount = 0
        self.sentimentScore = None
        self.messageTypes.append(messageType)
        Contact.contactsDict[name] = self


class Incoming:
    def __init__(self):
        self.timeStamps = []
        self.texts = []
        self.avgLag = None
        self.avgLagText = None
        self.stdLag = None
        self.lagTimes = None
        self.wordsCount = None
        self.tokenized = []
        self.sentimentScore = None
        self.positiveTexts = []
        self.neutralTexts = []
        self.negativeTexts = []


class Outgoing:
    def __init__(self):
        self.timeStamps = []
        self.texts = []
        self.avgLag = None
        self.avgLagText = None
        self.stdLag = None
        self.lagTimes = None
        self.wordsCount = None
        self.tokenized = []
        self.sentimentScore = None
        self.positiveTexts = []
        self.neutralTexts = []
        self.negativeTexts = []


# places text in correct list depending on whether message is incoming or outgoing
def textBin(contact, messageType, message, date):
    if messageType == 1:
        contact.incoming.texts.append(message)
        contact.incoming.timeStamps.append(date)
    if messageType == 2:
        contact.outgoing.texts.append(message)
        contact.outgoing.timeStamps.append(date)


def sortTimes(incomingTimes, outgoingTimes):
    times = incomingTimes
    times.extend(outgoingTimes)
    return sorted(times, key=itemgetter(0))


# updates contact object with lag times
def calculateLag(contact):
    incomingLags = []
    outgoingLags = []
    incomingTimes = [(inTime, True) for inTime in contact.incoming.timeStamps]
    outgoingTimes = [(outTime, False) for outTime in contact.outgoing.timeStamps]
    sortedTimes = sortTimes(incomingTimes, outgoingTimes)
    outgoingLag = None
    incomingLag = None
    # cases for length of zero
    if sortedTimes is None:
        print(contact.name)
        return incomingLag, outgoingLag
    for i, (timeStamp, isIncoming) in enumerate(sortedTimes):
        if i == 0:
            # stores message type of previous message
            startState = isIncoming
            startTime = timeStamp
        else:
            # check to see if reply
            if isIncoming != startState:
                lag = timeStamp - startTime
                if lag < 86400000:
                    if isIncoming:
                        incomingLags.append(lag)
                    else:
                        outgoingLags.append(lag)
                # switch state
                startState = isIncoming
                startTime = timeStamp

    if len(outgoingLags) != 0 and len(incomingLags) != 0:
        outgoingLag = utils.averageTime(outgoingLags)
        incomingLag = utils.averageTime(incomingLags)
        contact.incoming.lagTimes = incomingLags
        contact.outgoing.lagTimes = outgoingLags
        contact.incoming.avgLag = incomingLag
        contact.outgoing.avgLag = outgoingLag
        contact.incoming.avgLagText = utils.convertSeconds(incomingLag)
        contact.outgoing.avgLagText = utils.convertSeconds(outgoingLag)

    # variance calculations must have at least two elements
    if len(outgoingLags) >= 2 and len(incomingLags) >= 2:
        stdOutgoingLag = utils.stdDevTime(outgoingLags)
        stdIncomingLag = utils.stdDevTime(incomingLags)
        contact.incoming.stdLag = stdIncomingLag
        contact.outgoing.stdLag = stdOutgoingLag


def extractWords(texts):
    wordCount = 0
    tokenized = []
    for text in texts:
        tokens = utils.getTokens(text)
        wordCount += len(tokenized)
        tokenized.append(tokens)
    return wordCount, tokenized


def calculateSentiment(contact):
    outgoingSentimentScore = 0
    incomingSentimentScore = 0
    incomingTexts = contact.incoming.tokenized
    outgoingTexts = contact.outgoing.tokenized
    incomingRegularTexts = contact.incoming.texts
    outgoingRegularTexts = contact.outgoing.texts
    clf = joblib.load('main/textAnalysis/LogisticRegression().pkl')
    scaler = joblib.load('main/textAnalysis/Standard scaler.pkl')

    if len(incomingTexts) == 0 or len(outgoingTexts) == 0 or contact.textCount == 0:
        contact.incoming.sentimentScore, contact.outgoing.sentimentScore, contact.sentimentScore = 0, 0, 0
        return

    # calculate sentiment score and separate texts according to sentiment
    for i, message in enumerate(incomingTexts):
        isPositive, confidence = messageAnalysis.isPositive(message, clf, scaler)
        if isPositive:
            incomingSentimentScore += 1
            contact.incoming.positiveTexts.append((incomingRegularTexts[i], message, confidence))
        else:
            contact.incoming.negativeTexts.append((incomingRegularTexts[i], message, confidence))
    for i, message in enumerate(outgoingTexts):
        isPositive, confidence = messageAnalysis.isPositive(message, clf, scaler)
        if isPositive:
            outgoingSentimentScore += 1
            contact.outgoing.positiveTexts.append((outgoingRegularTexts[i], message, confidence))
        else:
            contact.outgoing.negativeTexts.append((outgoingRegularTexts[i], message, confidence))

    totalSentimentScore = int(((incomingSentimentScore+outgoingSentimentScore)/contact.textCount)*100)
    outgoingSentimentScore = (outgoingSentimentScore/len(outgoingTexts))*100
    incomingSentimentScore = (incomingSentimentScore/len(incomingTexts))*100
    contact.incoming.sentimentScore = int(incomingSentimentScore)
    contact.outgoing.sentimentScore = int(outgoingSentimentScore)
    contact.sentimentScore = totalSentimentScore


def organizeContact(contactObj):
    contactHistory = contactObj
    incomingTexts = contactHistory.incoming.texts
    outgoingTexts = contactHistory.outgoing.texts

    incomingWords, incomingTokenized = extractWords(incomingTexts)
    contactHistory.incoming.wordsCount = incomingWords
    # add each outgoing tokenized text to outgoing tokenized instance
    for outgoingT in utils.processTextChain(incomingTokenized):
        contactHistory.incoming.tokenized.append(outgoingT)

    outgoingWords, outgoingTokenized = extractWords(outgoingTexts)
    contactHistory.outgoing.wordsCount = outgoingWords
    # add each incoming tokenized text to incoming tokenized instance
    for incomingT in utils.processTextChain(outgoingTokenized):
        contactHistory.outgoing.tokenized.append(incomingT)


# everything in this function is original, although I gained insight on file structure
# from https://github.com/KermMartian/smsxml2html/blob/master/smsxml2html.py
def readTextsfromXml(xmlPath):
    tree = ET.parse(xmlPath)
    root = tree.getroot()
    startTime = t.time()

    # extract contacts from xml
    for child in root:
        if child.tag == 'sms':

            time = int(child.attrib['date'])  # Epoch timestamp
            messageType = int(child.attrib['type'])  # 1 = incoming, 2 = outgoing
            name = child.attrib['contact_name']
            body = child.attrib['body']

            # add text to contact
            if name in Contact.contactsDict:
                contact = Contact.contactsDict[name]
                contact.messageTypes.append(messageType)
            # initialize new contact
            else:
                contact = Contact(name, messageType, time)
                incoming = Incoming()
                outgoing = Incoming()
                contact.outgoing = outgoing
                contact.incoming = incoming
            contact.textCount += 1
            textBin(contact=contact, messageType=messageType, message=body, date=time)
    contacts = Contact.contactsDict
    print(contacts)
    print(t.time() - startTime)
    return contacts
