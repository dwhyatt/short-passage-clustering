#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# command line usage: english_short_passage_clusering.py text_to_cluster
# options:
# intialize: centroids = makeCentroidClusters(text, keywords = ['river', 'valley'])  (Keyword parameter is optional)
    
    # clusters = centroids.hardClusterLoop()
    # clusters = centroids.softClusterLoop()
    # clusters = centroids.multiloop()
    # clusters = centroids.taggedLoop()

# each function can also take an optional parameter to change the default clustering thresholds for the cosine similarity threshold.


import os
import sys
import re
import math
# import nltk
from nltk.collocations import *
from nltk.stem.wordnet import WordNetLemmatizer; lem = WordNetLemmatizer()
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from operator import itemgetter
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

##### if clustering on tagged data, the following should be uncommented
import nltk.data
from nltk.tag import stanford
# java_path = "/usr/bin/java"
# os.environ['JAVAHOME'] = java_path
# stanford_path = ('/Applications/stanford-postagger-full-2014-08-27/')




class makeCentroidClusters:  

    def __init__(self, text, keywords = []):  
        self.text = text
        self.clusters = [] 
        self.sentence_clusters = []
        self.keywords = keywords
        self.stopwords = ['a','able','about','across','after','all','almost','also','am','among', 'an',
             'and','any','are','as','at','be','because','been','but','by','can',
             'cannot','could','dear','did','do','does','either','else','ever','every',
             'for','from','get','got','had','has','have','he','her','hers','him','his',
             'how','however','i','if','in','into','is','it','its','just','least','let',
             'like','likely','may','me','might','most','must','my','neither','no','nor',
           'not','of','off','often','on','only','or','other','our','own','rather','said',
             'say','says','she','should','since','so','some','than','that','the','their',
             'them','then','there','these','they','this','tis','to','too','twas','us',
             'wants','was','we','were','what','when','where','which','while','who',
             'whom','why','will','with','would','yet','you','your', '.', ',', ';', ':'
             '1','2','3','4','5','6','7','8','9', '<new-para>', 'furthermore', 'therefore'
             'although', 'finally', 'conclusion', 'although', ';', ':', 'though', 'such',
             'this', 'that', 'these', 'those']
     

    def tokenize(self, text):
        
        list_of_sentences= sent_tokenize(text)
        return list_of_sentences


    def getCentroid(self, sentenceCluster):

        textCluster = []

        for sentence in sentenceCluster:
            for word in sentence:
                textCluster.append(word)
     
        wordFrequency = nltk.FreqDist(textCluster)
      
        sent = 0
        sentenceFrequency = {}
        for word in wordFrequency.keys():
            for sentence in sentenceCluster:
                sent += 1
                if word in sentence:
                    if word not in sentenceFrequency:
                        sentenceFrequency[word] = 1
                    else: 
                        sentenceFrequency[word] += 1

        scores = []
       
        for word in wordFrequency.keys():
            this_word_score = wordFrequency[word] * (1 + math.log(sent/sentenceFrequency[word]))
            scores.append([word, this_word_score])

        centroid = []   
        sorted_scores = sorted(scores, key=itemgetter(1), reverse = True)
        
        
        if len(sorted_scores) < 11:  ### should be reset to < 11, most likely.  Experimenting here a bit with length. 
            for pair in sorted_scores:
                centroid.append(pair[0])
        else:
            sorted_scores = sorted_scores[:10]
            for pair in sorted_scores:
                centroid.append(pair[0])

        return centroid


    def getCentroidScores(self, centroids, sentence_toMatch):
        
        which_centroid = []

        this_sent = ' '.join(sentence_toMatch)

        if any(t.isalpha() for t in this_sent):

            documents = []
            documents.append(this_sent)

            for centroid in centroids:
                if centroid:
                    centroid = ' '.join(centroid) 
                    documents.append(centroid)
            
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

            sim = (cosine_similarity(tfidf_matrix[0:1], tfidf_matrix))
        
            for s in sim:
                return s
           

    def getSentenceToMatch(self, sentence_toMatch):

        sentence_toMatch = [x.lower() for x in sentence_toMatch if x.lower()] ## not in self.keywords]
        sentence_toMatch = [lem.lemmatize(x) for x in sentence_toMatch if x not in self.stopwords and x not in self.keywords]  ##### do stuff here  #### sentence_toMatch is already a list
        sentence_toMatch = [x for x in sentence_toMatch if not x.isdigit()]
        sentence_toMatch = [x for x in sentence_toMatch if '-' not in x]
        sentence_toMatch = [x for x in sentence_toMatch if x]

        if not sentence_toMatch:
            return ['NOTHING']
        else:
            return sentence_toMatch


    def hardClusterLoop(self, threshold = .1, returning = True, matched = []): #### so that when conll != None we can skip write to "sentence"

      
        clusters = self.clusters
        sentence_clusters = self.sentence_clusters
      
        # number_of_sentences = 0

        list_of_sentences = self.tokenize(text)  ### this should be in the class initialization I think.


        for sentence in list_of_sentences:
            sentence = word_tokenize(sentence)

            # number_of_sentences += 1
           
            sentence_toMatch = self.getSentenceToMatch(sentence)  #### this could refined so as not to redo this.
           
            if sentence_toMatch in matched:
                pass

            else:

                centroids = []

                if clusters:

                    cluster_count = 0
                    for c in clusters:
                        centroids.append(self.getCentroid(c))
                        cluster_count += 1

                    this_matrix = []
                    this_matrix.append(self.getCentroidScores(centroids, sentence_toMatch))
                 
                    candidates = []
                    
                    for v in this_matrix:
                        if v is not None:
                            for score in v[1:]:  
                                candidates.append(score)
                    
                    if candidates:
                       
                        best_score = max(candidates)
                        best_cluster = candidates.index(max(candidates))

                        if best_score > threshold:    
                            clusters[best_cluster].append(sentence_toMatch)
                            sentence_clusters[best_cluster].append(sentence)
                         
                        else:
                            
                            clusters.append([sentence_toMatch])
                            sentence_clusters.append([sentence])

                    if not candidates:
                        pass
                      

                if not clusters:   
                    clusters.append([sentence_toMatch])
                    sentence_clusters.append([sentence])
                  

            centroids = []
            sentence_toMatch = []

        if returning == True:

            return self.getResults()

    def softClusterLoop(self, threshold = .04, returning = True, matched = []): 

        clusters = self.clusters
        sentence_clusters = self.sentence_clusters
        # number_of_sentences = 0

        list_of_sentences = self.tokenize(text) 


        for sentence in list_of_sentences:
            sentence = word_tokenize(sentence)

            # number_of_sentences += 1
           
            sentence_toMatch = self.getSentenceToMatch(sentence)  
           
            if sentence_toMatch in matched:
                pass

            else:

                centroids = []

                if clusters:

                    cluster_count = 0
                    for c in clusters:

                        centroids.append(self.getCentroid(c))
                        cluster_count += 1

                    this_matrix = []
                    this_matrix.append(self.getCentroidScores(centroids, sentence_toMatch))
                 
                    candidates = []
                    
                    for v in this_matrix:
                        if v is not None:
                            for score in v[1:]:  ### so we don't compare it to itself
                                candidates.append(score)

           
                    appended = False
                    if candidates:
                       
                        n = 0 
                        for c in candidates:
                   
                            if c > threshold:    
                                appended = True
                    
                                clusters[n].append(sentence_toMatch)
                                sentence_clusters[n].append(sentence)
                                n += 1
                            else:
                                n += 1
                           
                         
                        if appended == False:
                            
                            clusters.append([sentence_toMatch])
                            sentence_clusters.append([sentence])
                           

                    if not candidates:
                        pass
                       

                if not clusters:   
                    
                    clusters.append([sentence_toMatch])
                    sentence_clusters.append([sentence]) 

            centroids = []
            sentence_toMatch = []

        if returning == True:

            self.getResults()

    def getNounsAndVerbs(self, sentence):

        nouns_and_verbs = []

        for pair in sentence:
            word = pair[0]; tag = pair[1]
            tag = tag.lower()
            word = word.lower()
            if tag.startswith('n'):
                nouns_and_verbs.append(word)
            elif tag.startswith('v'):
                nouns_and_verbs.append(word)
            else:
                pass

        return self.getSentenceToMatch(nouns_and_verbs)

    def tagEnglish(self):

        java_path = "/usr/bin/java"
        os.environ['JAVAHOME'] = java_path
        stanford_path = ('/Applications/stanford-postagger-full-2014-08-27/')

        tag_english = stanford.POSTagger(stanford_path + 'models/english-bidirectional-distsim.tagger', stanford_path + 'stanford-postagger.jar', encoding = 'utf-8')

        sentences = nltk.sent_tokenize(self.text) 
        tokens = [nltk.word_tokenize(sent) for sent in sentences] 
        tagged = tag_english.batch_tag(tokens)  #### this needs to be changed to sent_tag for newer versions of NLTK
        
        return tagged

    def taggedLoop(self, threshold = .04, returning = True, matched = []): 

        clusters = self.clusters
        sentence_clusters = self.sentence_clusters

        tagged = self.tagEnglish()
        

        for sentence in tagged:
            
            words_only = [pair[0] for pair in sentence]
            
            nouns_and_verbs = self.getNounsAndVerbs(sentence) 

            centroids = []

            if clusters:

                cluster_count = 0
                for c in clusters:

                    centroids.append(self.getCentroid(c))
                    cluster_count += 1

                this_matrix = []
                this_matrix.append(self.getCentroidScores(centroids, nouns_and_verbs))
             
                candidates = []
                
                for v in this_matrix:
                    if v is not None:
                        for score in v[1:]:  
                            candidates.append(score)

       
                appended = False
                if candidates:
                   
                    n = 0 
                    for c in candidates:
               
                        if c > threshold:    
                            appended = True
                
                            clusters[n].append(nouns_and_verbs)
                            sentence_clusters[n].append(words_only)
                            n += 1
                        else:
                            n += 1
                       
                     
                    if appended == False:
                        
                        clusters.append([nouns_and_verbs])
                        sentence_clusters.append([words_only])
                       

                if not candidates:
                    pass
                   

            if not clusters:   
                
                clusters.append([nouns_and_verbs])
                sentence_clusters.append([words_only]) 

            centroids = []
            

        if returning == True:

            self.getResults()

    
    def getResults(self):     #### this is currently being modified to return only stanford pipeline sentences, in future in would make sense
        

        clusters = self.clusters
        sentence_clusters = self.sentence_clusters
    
      
        centroid_list = []
        for c in clusters:
            centroid_list.append(self.getCentroid(c))

        centroid_list = ((len(c), c) for c in centroid_list)
        print('number of centroids', len(clusters))
        print('\n')


        results = zip(centroid_list, sentence_clusters)

        for r in results:
            cluster_keywords = r[0]
            cluster_sentences = r[1]
            print(cluster_keywords)
            #print('cluster_sentences ', cluster_sentences)
            for cl in cluster_sentences:
                cl = ' '.join(cl)
                print(cl)
            print('\n')
    


    def multiloop(self):   

        
        sentence_clusters = self.sentence_clusters
        clusters = self.clusters

        print('here!')
        print(len(sentence_clusters), len(clusters))
        threshold = .5                                                   \

        self.hardClusterLoop(threshold, False)  

        matched = []
       
        while threshold > .1:
            
            clusters_removed = []
            sentences_removed = []

            i = 0
            while i < len(clusters):

                if len(clusters) != len(sentence_clusters):
                    print('orginal clusters off')
                i += 1
                assert len(clusters) == len(sentence_clusters)
            

            i = 0
            while i < len(clusters):
                
                cl = 0; s = 0
                if len(clusters[i]) == 1:
                    clusters_removed.append(clusters[i])
                    cl = clusters[i]
                    clusters.remove(clusters[i])
                    
                if len(sentence_clusters[i]) == 1:
                    sentences_removed.append(sentence_clusters[i])
                    s = sentence_clusters[i]
                    sentence_clusters.remove(sentence_clusters[i])

                if cl and not s:
                    print('warning: alingment off')
                    #clusters.append(['HOLDING CLUSTER'])
                    assert len(clusters) == len(sentence_clusters)

                if s and not cl:
                    print('warning: alignment off')
                    #sentence_clusters.append(['HOLDING CLUSTER'])
                    assert len(clusters) == len(sentence_clusters)

                i += 1
        
            
            for c in clusters:
                for sent in c:
                    matched.append(sent)

            print('threshold', threshold, 'number of matched', len(matched), 'current length of clusters', len(clusters), 'current length of sentence clusters', len(sentence_clusters))
            threshold -= .05   
            self.hardClusterLoop(threshold, False, matched)
            matched = []

        else:
            
            threshold = .1
            print('threshold', threshold, 'number of matched', len(matched), 'current length of clusters', len(clusters))
            for c in clusters:
                if len(c) == 1:
                    clusters.remove(c)

            for s in sentence_clusters:
                if len(s) == 1:
                    sentence_clusters.remove(s)
           
            for c in clusters:
               
                for sent in c:
                    matched.append(sent)

            
            self.hardClusterLoop(threshold, True, matched)




if  __name__ =='__main__': 

    
    infile = sys.argv[1]
    text = open(infile, 'r').read()

    centroids = makeCentroidClusters(text, keywords = ['river', 'valley'])
    clusters = centroids.taggedLoop()
    centroids = makeCentroidClusters(text) 
    clusters2 = centroids.hardClusterLoop()
    centroids = makeCentroidClusters(text)
    cluster3 = centroids.multiloop()
    #print(clusters)
    #print(clusters2)


    
    

    







