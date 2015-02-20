# short-passage-clustering
A clustering program based on Seno &amp; Nunes (2008)

The original version of this code was written for the Indiana University project "Linguistic Corpus and Ontology for Comparative Analysis of Networks", with PI Armando Razo & co-PI Markus Dickinson, funded by the IU Faculty Research Support Program (FRSP).

This program is based on the text clustering algorithm described in: Seno, E. R. M., & Nunes, M. D. G. V. (2008). SiSPI: A Short-Passage Clustering System. ICMC-USP.  In a nutshell, this algorithm builds sentence clusters in a single pass by making a centroid of the most important words in each cluster, and then uses a cosine similarity score threshold to add subsequent sentences to the cluster; i.e., the first sentence of the text forms the first cluster, the centroid is calculated, the next sentence is either added to the first cluster, or forms its own cluster, the centroid(s) are (re)calculated, etc., until all sentences are in clusters.       

A version of the algorithm described in Seno & Nunes (2008) is a hard-clustering algorithm, and so is implemented here as the method hardClusterLoop.  There is also a soft-clustering option in which sentences are added to all clusters to which they match above a set threshold, rather than being added only once to the cluster with the highest similarity score.      

Additional options you can try include the multiloop method, which begins with a high threshold score to build clusters, then in each iteration removes all singleton clusters (clusters with one sentence only), and tries again in clustering these sentences, lowering the threshold score at each iteration.  In some cases this might produce preferable results, and is less susceptible to ordering affects.  Also, there is an option to part-of-speech tag the input text, and then to cluster considering only those words which are tagged as verbs or nouns, as they are typically the critical words in capturing the content of a sentence.  

If you have a list of keywords you have previously identified as being important for the text (via a topic-modeling tool, for instance), this can be included as a parameter when you initialize the class, in which case these words are ignored in building centroid clusters.  This may be useful in providing additional information about the text apart from the previously identified keywords, such as in cases where the keywords might lead to coarser clustering than desired (for instance, if you wanted separate clusters for Irish and American folk music, and 'folk' and 'music' were known keywords).

A list of English stopwords are included (which are ignored), and all words are lemmatized with an English-language lemmatizer.  For best results in a language other than English, a list of stopwords for that language should be swapped in, along with a lemmatizer specific to the language.

To run this program, nltk (http://www.nltk.org/install.html) and scikit-learn must be installed (available from http://scikit-learn.org/stable/install.html).  Also, the code here uses the stanford tagger (http://nlp.stanford.edu/software/tagger.shtml), and the correct path to the stanford POS jar file should be inserted in the tagEnglish function.  There are also other options available for POS tagging in nltk.  Finally, please note that the code in the tagEnglish function uses the 'batch_tag' method, which may have to be changed to 'sent_tag' in recent versions of NTLK. 


Usage:

short_passage_clustering.py text_to_cluster

clusters = makeCentroidClusters(text, keywords = ['river', 'valley'])  (Keyword parameter is optional)

clusters.hardClusterLoop()
clusters.softClusterLoop()
clusters.multiloop()
clusters.taggedLoop()

each function can also take an optional parameter to change the default clustering thresholds for the cosine similarity threshold.     

centroids.hardClusterLoop(threshold = .2)