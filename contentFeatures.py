# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import json
import pprint
import numpy as np
from math import log


            
def get_word_space(s1,s2):
    "Returns distinct words from sentance s1 and s2"
    return list(set(s1.lower().split()).union(set(s2.lower().split())))

def get_sentence_vector(sentence,word_list):
    "Returns vector representation of a sentence based on the number of occurences of a word in that sentence"
    vector = []
    allWords = sentence.lower().split()
    for word in word_list:
        vector.append(allWords.count(word))
    return vector
        

def get_cosine_similarity(a,b): 
    "Returns the cosine similarity of the two vectors"
    return np.dot(a,b)/(np.sqrt(np.dot(a,a)) * np.sqrt(np.dot(b,b)))

def termFrequency(document):
    "Returns the normalized term frequency for each term in the document"
    normalized = document.lower().split()
    tf = dict()
    for term in normalized:
        tf[term] = normalized.count(term.lower()) / float(len(normalized))
    return tf

def inverseDocumentFrequency(term, allDocuments):
    "Returns inverse document frequencies of each term"
    numDocumentsWithThisTerm = 0
    for doc in allDocuments:
        if term.lower() in doc.lower().split():
            numDocumentsWithThisTerm = numDocumentsWithThisTerm + 1
 
    if numDocumentsWithThisTerm > 0:
        return 1.0 + log(float(len(allDocuments)) / numDocumentsWithThisTerm)
    else:
        return 1.0

def tfidf(queryTf,idf):
    "Returns the tfidf of every term in the query"
    allWords = list(idf.keys())
    queryWords = queryTf.keys()
    vector = []
    for term in allWords:
        if term in queryWords:
            vector.append(queryTf[term]*idf[term])
        else: 
            vector.append(0)
    return vector
        
        
def utteranceSimilarity(query, sentence):
    "returns the cosine similarity between sentence1 and sentence2"
    # get the normalised term frequencies
    tf1 = dict()
    tf2 = dict()
    tf1 = termFrequency(query) # frequency of each word in the query
    tf2 = termFrequency(sentence) # frequency of each word in the query in sentence
    
    # get the inverse document frequencies for each word
    store=dict() # inverse document frequency of every distinct word
    for word in get_word_space(query,sentence):
        store[word] = inverseDocumentFrequency(word,[query,sentence])
        
    # get the tfidf for each the two sentences
    a = tfidf(tf1,store)
    b = tfidf(tf2,store)
    
    #get the cosine similarities between the two sentences
    return get_cosine_similarity(a,b)

#def dialogueSimilarity(utterance, allUtterences):
    #"Returns the cosine similarity between one sentence and all sentences"
    

def containsQuestionMark(utterance):
    "Returns true if an utterance contains '?'"
    if "?" in utterance:
        return True
    return False

def containsDuplicateKeywords(utterance):
    "Returns true if an utterance contains 'same' or 'similar'"
    if 'same' in utterance or 'similar' in utterance:
        return True
    return False

def containsQuestionWords(utterance):
    "Returns a vector indicating the presence of what, where, when, why, who, how - in this order"
    hotVector = ['what', 'where', 'when', 'why', 'who', 'how']
    questionWords = np.zeros(6,int)
    for i in range(len(hotVector)):
        if hotVector[i] in utterance:
            questionWords[i] = 1;
    return questionWords
        
    
def getContentFeatures(startingUtterance, currentUtterance, allUtterances):
    "Returns the content features for an utterance"
    contentFeatures = []
    contentFeatures.append(utteranceSimilarity(startingUtterance, currentUtterance))
    contentFeatures.append(utteranceSimilarity(currentUtterance,allUtterances))
    contentFeatures.append(containsQuestionMark(currentUtterance))
    contentFeatures.append(containsDuplicateKeywords(currentUtterance))
    contentFeatures.append(containsQuestionWords(currentUtterance))
    return contentFeatures

def contentFeaturesTest():
    query = "life learning" 
    s1 = "The game of life is a game of everlasting learning?"
    s2 = "The unexamined life is not worth living"
    s3 = "Never stop learning"
    features = getContentFeatures(query,s1, s1 + '' + s2 + '' + s3)
    


