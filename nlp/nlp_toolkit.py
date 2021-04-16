#!/usr/bin/env python
# coding: utf-8

# # NLP Swiss Army Knife
# A number of functions desinged to give a low barrier of entry to lots of Natural Lanuguage Processing functions and display options.
#
# A typical use case would be being asked (as the Analytics point of contact) to analyse a spreadsheet of survey comments; a task that you do not want to devote weeks of effort to, but that it would be good to throw some cutting edge analytics at to showcase what we can do.
#
# Not designed to be used for a specific NLP assignment for deploying models to the cloud. This is for quick and dirty NLP on dataframes.

# ## What we are aiming for
# Some useful visualisations and interactive tools to help us get a sense about what is being talked about in a selection of (typicall short) comments. We will produce much simpler plots than shown below.
#
# ### Using Scattertext to show which words are associated between two groups or characteristics
#
# <img src="scattertext_large_plot.png">
#
# ### Using UMAP to reduce our knowledge of words into a 2D space to visualise clusters of similarity
# <img src="umap_large_plot.png">
#
# ### Using Tiddlywiki to produce an interactive tool to show word associations within our corpus
# <img src="tiddler_keyword_pic.png">
#

# ### And some wordclouds!
# <img src="wordcloud.png">

# ## How we will do it: Modifying Pandas with NLP functions
# We use the pandas_flavor package together with the @pandas_flavor.register_dataframe_method decorator to allow us to pipe NLP processing steps onto the dataframe
#
# For example:
# ```python
# df.nlp_extract_sentiment('comments').nlp_extract_entities('comments')
# ```
#
# The details and what NLP libraries are being used are hidden from the user, this makes it very quick to apply.

# ## Packages we need
#

# In[3]:


import pandas_flavor as pf
import pandas as pd
import matplotlib.pyplot as plt
from itertools import repeat
import itertools
import networkx as nx
import janitor as jn
import numpy as np
import json
import gensim
import pyLDAvis.gensim
import nltk
import spacy
from spacy.matcher import Matcher
import seaborn as sns
from wordcloud import WordCloud

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from functools import reduce
from scipy.stats import zscore

import scattertext as st
# import umap
# import umap.plot
from bokeh.plotting import output_file, save
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import pysnooper


#!python -m spacy download en_core_web_md #this needed to download the spacy language model


# ## Download and initialise the various NLP models
# Note we are using the medium NLP model which has word vectors which we will need later.

# In[4]:





# ## Register the dataframe methods
# They all have an 'nlp_' prefix as a common naming convention
#
# We will look into these individually on a simple dataset.
#

# In[5]:


@pf.register_dataframe_method
def nlp_apply_lda(df, column_name, target_column_name = 'lda_topics', topics_per_doc = 2, save_html=True, html_filename="LDA topics.html", column_format='list', **kwargs):
    """
    applies LDA to column (the column is the corpus, each row a document)

    **kwargs gets passed into gensim.models.ldamodel
    each document should be in gensim format - i.e. a list of strings, if it isn't just pass column_format='string'

    If you want to apply filter_extremes, do this earlier with nlp_filter_extremes()

    outputs: an interactive LDAVis html page (if save_html = True (default))
    also returns: the data-frame with a new column 'lda_topics' which is a dictionary of topic_id : topic_% pairs"""
    #build dictionary
    if column_format == 'string':
        df[column_name] = df[column_name].apply(string_to_list)

    dictionary = gensim.corpora.Dictionary(df[column_name])
    bow_corpus = [dictionary.doc2bow(text) for text in df[column_name]]
    lda_model = gensim.models.LdaModel(bow_corpus, id2word=dictionary, **kwargs)
    lda_corpus = lda_model[bow_corpus]

    lda_topics_df = pd.DataFrame()

    for i, row in enumerate(lda_corpus):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        temp_di = dict()
        for topic_rank, (topic_num, prop_topic) in enumerate(row):
            if topic_rank < topics_per_doc:  # => dominant topic
                temp_di.update({topic_num : round(prop_topic,4)})
            else:
                break
        lda_topics_df = lda_topics_df.append(pd.Series([temp_di]), ignore_index=True)

    lda_topics_df.columns = [target_column_name]
    df = pd.concat([df,lda_topics_df], axis=1)

    if save_html:
        vis_prep = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary)
        pyLDAvis.save_html(vis_prep, html_filename) #interactive HTML file from LDAVis

    return df

@pf.register_dataframe_method
def nlp_redact_entities(df,nlp_column_name:str,entity_types:list, target_column_name = 'redacted_text', replace_text = 'XX_redacted_XX'):
    """redacts certain entities, replacing the text with replace_text, useful for removing names in comments for example

    typical use would be entity_types = ['PERSON'] to ensure no names occur in survey comments. Uses SPACY entity types https://spacy.io/api/annotation#named-entities

    returns: the dataframe with a new column of the redacted text"""

    #go through each nlp item
    redacted_corpus = []
    for document in df[nlp_column_name]:
        redacted_corpus.append(" ".join([token.text if token.ent_type_ not in entity_types else replace_text for token in document]))
    df[target_column_name] = redacted_corpus
    return df

@pf.register_dataframe_method
def nlp_extract_sentiment(df, column_name, target_column_name = 'sentiment'):
    sid = nltk.sentiment.vader.SentimentIntensityAnalyzer()
    df[target_column_name] = df[column_name].map(lambda x: sid.polarity_scores(x)['compound'])
    return df

@pf.register_dataframe_method
def nlp_set_of_entities(df, column_name:str,entity_types:list, target_column_name = 'entity_sets'):
    """extracts certain entities into a new column as a set - from the extract entities (dictionary) column
    Repalces spaces with an underscore with entities more than one word long - useful for scattertext function

    Uses SPACY entity types https://spacy.io/api/annotation#named-entities

    returns: the dataframe with a new column of entities in a set"""

    #go through each nlp item
    entity_list = []
    for dic in df[column_name]:
        row_list = []
        for key in dic:
            if key in entity_types:
                #now dic[key] is a list of entities

                row_list = row_list + [item.replace(" ", "_") for item in dic[key]]
                #print("appending ",[item.replace(" ", "_") for item in dic[key]] )
        entity_list.append(row_list)
    df[target_column_name] = entity_list
    df[target_column_name] = df[target_column_name].map(set)
    return df

@pf.register_dataframe_method
def nlp_process_column(df, column_name, nlp_object, new_column_name='nlp'):
    """
    applies spacy nlp object to a column that contains text

    you will need to have imported spacy and previously created an nlp_object

    :return: returns dataframe with a new column that has nlp objects in it
    """
    df[new_column_name] = df[column_name].apply(nlp_object)
    return df

@pf.register_dataframe_method
def nlp_count_sentences(df, column_name, new_column_name = 'number_sentences'):
    """
    counts the number of sentences: column must contain spacy NLP objects

    The number of sentences can be used in conjucntion with the nlp_one_row_per_sentence method
    to return a dataframe which has one row per sentnece. Useful if you want to investigate text on a sentence by sentence basis

    :return: returns an integer column of the number of sentences
    """

    df[new_column_name] = df[column_name].apply(lambda x: len(list(x.sents)))
    return df

@pf.register_dataframe_method
def nlp_one_row_per_sentence(df, column_name = 'nlp', new_column_name = 'nlp_sentences', num_sentences = 'number_sentences'):
    """
    Produces a dataframe of one (NLP) sentence per row based on NLP objects in colum_name

    It requires nlp_count_sentences to be run before hand as this produces the num_sentences column

    :return: returns dataframe with one NLP sentence per row"""

    sentence_list = []
    for doc in df[column_name]:
        for sentence in doc.sents:
            sentence_list.append(sentence)
    df = df.loc[df.index.repeat(df[num_sentences])]
    df[new_column_name] = sentence_list #this is nlp object
    return df

@pf.register_dataframe_method
def nlp_extract_lemma_text(df, nlp_column, new_column_name = 'lemma_sentences', keep=[], token_nlp=False):
    """converts an NLP column into a sentence of lemma tokens, removing stop-words and punctuation and the lemma -PRON-

    useful for getting a version of each sentence for keyword extraction later. If the NLP object has more than one sentence,
    these will get combined into a single sentence, so consider using nlp_one_row_per_sentence to avoid this

    If token_nlp is True, rather than returning each token as a string, it returns it as a token, so keeps the vector embedding and entity data

    :return:returns a dataframe with a new column called 'lemma_sentences'
    """

    def punct_space_stop(token):
        """
        helper function to eliminate tokens
        that are pure punctuation or whitespace or stop words or pronouns
        """
        if token.text in keep or token.dep_ in keep or token.pos_ in keep:
            return False
        else:
            return token.is_punct or token.is_space or token.lemma_ == "-PRON-" or token.is_stop

    lemma_sents = []
    for doc in df[nlp_column]:
        lemma_sents.append(' '.join([token.lemma_ for token in doc if not punct_space_stop(token)]))
    df[new_column_name] = lemma_sents
    return df


#gets a simpler version of the comments to prepare for keyword extraction. Extracts the lemma for each
#token and removes stop words.

@pf.register_dataframe_method
def nlp_apply_topic(df,nlp_object,compare_column,topic_string, topic_column, master_topic_column='topics', threshold=0.8, comparison_type = 'nlp'):
    """
    Used to assign custom topic keywords. Uses either word_similarity (default), or just keyword matching

    comparision_type is one of ['nlp' , 'text' ,'keyword_match']

    'nlp' - you are comparing against a column that is a Spacy NLP object (i.e. already vectorised)
    'text' - you are comparing against a text column, this will take more time as NLP model is run against each row of text first
    'keyword_match' - you are looking for matching words, the other column will need to be a text column for this to work

    This returns a dataframe with 2 new columns, a topic_colum with the match score, this is similarity word vector
    matching or the % of number of words matched (if set_compare=True), use clean keywords if using set_compare (lemmas etc.) as it
    won't match 'word' with 'words' for example, perfect case matching only. With word similarity this is less important.
    You must of course have an nlp object that contains word vectors.
    The  master_topic_column is created to hold a  set of matching topics on the occasion this function is called multiple
    times in a sequennce on the same data-frame (the intention), then there is a column of all matching topics for a given row

    The function works well if used iteratively, try some topic_strings, then sort_values on the dataframe to see if there is a
    good match,adjust the threshold accordingly to only apply the topic to matching sentences. Then move on to the next topic to
    assign.

    :topic_string: this is a string separated out by spaces, if passing a list of words (for set_compare=T), just write them with spaces
    e.g. "item1 item2 item3"

    :returns:dataframe with 2 new columns, a score (topic_column) and set of matches (master_topic_column)"""

    def assign_topic(df, topic_column, threshold, master_topic_column):
        """helper function to assign matching topics to the dataframe in the correct place,
        sorts out weird behaviour with NAs and sets"""
        mask = df[topic_column] > threshold
        items = df.loc[mask,master_topic_column].fillna("").map(set)
        items.apply(lambda x: x.add(topic_column))
        df.loc[mask,master_topic_column] = items
        return df

    topic_sim = []
    topic_nlp = nlp_object(topic_string)
    if master_topic_column not in df.columns:
        df[master_topic_column] = list(repeat(set(), len(df))) # create empty set

    if comparison_type == "keyword_match":
        #we are expecting topic_string and compare_columns to both be sets
        if type(topic_string) == str:
            topic_string = set(topic_string.split(" ")) # str -> set
        number_items_topic = len(topic_string)
        for a_set in create_set_from_series(df[compare_column]):
            #print(a_set, topic_string)
            number_not_matched = len(topic_string.difference(a_set))
            topic_sim.append(1 - number_not_matched/number_items_topic)
    elif comparison_type == "text":
        #print("printing master topic not in",df[master_topic_column])
        df[compare_column] = create_clean_text(df[compare_column])
        #if type(df[compare_column][0]) == set:
            #df[compare_column] = df[compare_column].map(lambda x: " ".join(list(x)))
            #print("compare_column is a set", df[compare_column])
        for sent in df[compare_column]:
            topic_sim.append(topic_nlp.similarity(nlp_object(sent)))
    elif comparison_type == "nlp":
        for document in df[compare_column]:
            topic_sim.append(topic_nlp.similarity(document))
    df[topic_column] = topic_sim
    df = assign_topic(df, topic_column, threshold, master_topic_column) # add the topic word to the set of topics for each row
    return df

@pf.register_dataframe_method
def nlp_apply_tfidf(df, column_name, new_column_name ='keywords', threshold = 0.4, ngram_range=(1,1), top_n = 0, column_format = 'string', append=False):
    """extracts keywords based on TFIDF for a given threshold, can be applied to any string

    threshold: the TFIDF score to be greater than to extract a keyword
    top_n: if this is greater than 0 it will return the top n keywords (if fewer make the threshold then fewer are returned)

    if column_format is 'list' it will convert to a string first

    :returns: dataframe with sets of keywords in a column"""

    if column_format == 'list':
        df[column_name] = df[column_name].apply(list_to_string)

    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    #print(df[column_name])
    vectors = vectorizer.fit_transform(df[column_name])
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df2 = pd.DataFrame(denselist, columns=feature_names)
    if top_n > 0:
        dfagg = (df2.reset_index() #this gets top keywords for each index
            .melt(id_vars='index')
            .groupby(['index','variable'])
            .agg(sum)
            .filter_on("value >"+str(threshold)) # what value to filter on with TFIDF score
            .value
            .groupby(level=0, group_keys=False).nlargest(top_n)
            .reset_index()
        )
    else:
        dfagg = (df2.reset_index() #this gets top keywords for each index
            .melt(id_vars='index')
            .groupby(['index','variable'])
            .agg(sum)
            .filter_on("value >"+str(threshold)) # what value to filter on with TFIDF score
            .value
            .reset_index()
        )
    df_key = dfagg.groupby('index').variable.apply(lambda x: set(x))
    df[new_column_name] = df_key.fillna("").apply(set)
    df[new_column_name] = df[new_column_name].fillna("").apply(set)
    #df[new_column_name] = df_key
    #print(df_key)
    if append == True:
        return pd.concat([df,df2],axis=1)
    else:
        return df


@pf.register_dataframe_method
def nlp_extract_bigrams(df, unigram_column, target_column_name = "bigram_column", **kwargs):
    """
    extracts bigrams using the gensim Phrases models

    You can pass more **kwargs into the Phrases function e.g. min_count, scoring, threshold

    """
    bigram = gensim.models.Phrases(df[unigram_column], **kwargs)
    bigram = gensim.models.phrases.Phraser(bigram)
    df = df.transform_column(unigram_column, lambda x: bigram[x], dest_column_name = target_column_name)
    return df

@pf.register_dataframe_method
def nlp_extract_entities(df, nlp_column_name='nlp', new_column_name='entities'):
    """
    extracts entities from an NLP object column into a dictionary column

    :return: returns dataframe with a new column containing a list of dictionaries of entities"""
    def entity_dictionary(nlp):
        entity_dict = {}
        for entity in nlp.ents:
            if entity.label_ not in entity_dict.keys():
                entity_dict[entity.label_] = [entity.lower_]
            else:
                entity_dict[entity.label_].append(entity.lower_)
        return entity_dict

    df[new_column_name] = df[nlp_column_name].apply(entity_dictionary)
    return df

@pf.register_dataframe_method
def nlp_extract_matches(df, nlp_column, matcher_object, patterns, new_column_name = 'match', lemma=True):
    """
    extracts lemma(default) or raw text based on an NLP matcher object and patterns

    You need to have delcared a Matcher object e.g. matcher = Matcher(nlp.vocab)
    patterns can be 1 or a list of patterns

    :returns: dataframe with column of text (lemma(default) or raw) of the matches based on patterns (list of Spacy patterns)"""

    matcher_object.add(new_column_name, None, *patterns)

    text_matches = []
    for doc in df[nlp_column]:
        sentences = []
        for sentence in doc.sents:
            matches = matcher_object(sentence)

            for id, start, end in matches:
                #print("pattern:",sentence[start:end], "TEXT ", sentence.text)
                short_sentence = []

                for token in sentence[start:end]:
                    if lemma:
                        short_sentence.append(token.lemma_)
                    else:
                        short_sentence.append(token.text)
                #print("short sentence to append", short_sentence)
                sentences.append(" ".join(short_sentence))
        text_matches.append([sentences])
    #print(text_matches)

    df[new_column_name] = pd.DataFrame(text_matches)
    return df

@pf.register_dataframe_method
def nlp_extract_pos(df, nlp_column, token_pos:list, new_column_name = 'pos', lemma=True):
    """
    extracts lemma(default) or raw text based on parts of speech or dependencies

    It will only extract tokens with specific parts of speech, such as ROOT, nsubj or VERB listed in token_pos

    :returns: dataframe with column of text (lemma(default) or raw) of list of matches (per sentence)
    based on token_pos (list of Spacy pos or dep names)"""

    text_matches = []
    for doc in df[nlp_column]:
        matches = []
        for token in doc:
            if token.pos_ in token_pos:
                if lemma:
                    matches.append(token.lemma_)
                else:
                    matches.append(token.text)
        text_matches.append([matches])

    df[new_column_name] = pd.DataFrame(text_matches)
    return df

@pf.register_dataframe_method
def nlp_filter_extremes(df, text_column, target_column_name = 'filtered_text', **kwargs):
    """
    converts a gensim document (list of strings) into a filtered equivalent (removing certain words)

    **kwargs will go into the gensim dictionary.filter_extremes() function (no_below, no_above, keep_n) etc.
    """
    dictionary = gensim.corpora.Dictionary(df[text_column])
    dictionary.filter_extremes(**kwargs)

    def remove_filtered_tokens(my_list):
        """
        helper function: only keeps words that have not been filtered out by the dictionary filter_extremes function"""
        return [word for word in my_list if word in dictionary.values()]

    df[target_column_name] = df[text_column].apply(remove_filtered_tokens)
    return df

@pf.register_dataframe_method
def nlp_sub_word(df, column_name, pattern, replace, flags=re.IGNORECASE):
    """A find and replace function for replacing words, ignoring punctuation and spaces
    This is because Excel find and replace can't do this easily
    This function finds your pattern string and replaces it with replace
    your pattern string will be a word and this searches for that word - in between anything other than letters and numbers

    returns the dataframe"""
    df = df.transform_column(column_name, lambda x: re.sub(r'(^|[^\w])' + pattern + r'([^\w]|$)',r'\1' + replace + r'\2', x,flags=flags))
    return df

@pf.register_dataframe_method
def nlp_text_length(df, text_column, new_column_name = 'comment_length'):
    df = df.transform_column(column_name = text_column,
                        function = lambda x: int(len(str(x))),
                        dest_column_name = new_column_name)
    return df


# ## Create some helpful functions
# For cleaning text and visualising the results.
#
# Much of this code is for creating a co-occurence graph in the TiddlyWiki JSON format. (See www.tiddlywiki.com) with the tiddlymap plugin.

# In[338]:



@pysnooper.snoop('log.txt')
class KeywordGraph(nx.Graph):

    def __init__(self,data,node_column,attributes:dict, nlp_object = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = data
        self.node_column = node_column
        self.attributes = attributes
        self.build_graph()
        if nlp_object != None:
            self.add_spacy_language_model(nlp_object)
        else:
            self.nlp_object = None

    def add_spacy_language_model(self,nlp):
        self.nlp_object = nlp

    def _keyword_counts(self):
        #returns dictionary of keyword : count, ser is a Series of keywords as sets
        self.data[self.node_column] = self.data[self.node_column].apply(set)
        def initialise_keyword_di(ser):
            self.data[self.node_column] = self.data[self.node_column].fillna("")
            #returns a dictionary with a key of every keyword and a value of 0, takes in dataframe series of sets of keywords
            all_keys = reduce(set.union, self.data[self.node_column])
            keyword_d = {}
            for key in all_keys:
                keyword_d[key] = 0
            return keyword_d

        def update_keyword_dict(keyword_d, set_of_keywords): #updates value in dictionary
            for keyword in set_of_keywords:
                keyword_d[keyword] += 1
            return keyword_d

        keyword_d = initialise_keyword_di(self.data[self.node_column])
        for keyword_set in self.data[self.node_column]:
            keyword_d = update_keyword_dict(keyword_d, keyword_set)
        return keyword_d

    def add_graph_attribute(self,column,attribute_name):
        for _index in self.data.index:
            entry = self.data.loc[_index,column].replace("'","\'")
            nodes = self.data.loc[_index,self.node_column]

            #go through each node and find matches in the graph to add the entry to
            for node in nodes:
                if attribute_name in self.nodes[node]: # there is already another entry here in this attribute
                    self.nodes[node][attribute_name].append(entry)

                else: #this is a new entry
                    self.nodes[node][attribute_name] = [entry]
    #apply extra attributes to the graph
    def build_graph(self):
        #define our graph
        #keyword_graph = nx.Graph()
        self.keyword_counts = self._keyword_counts() #get keyword counts of keyword : count

        for _key in self.keyword_counts:
            self.add_node(_key, node_count = self.keyword_counts[_key]) #add keyword counts to the graph
        #add nodes from each keyword

        #add edges of keyword co_occurence
        for _set in self.data[self.node_column]:
            combinations = itertools.combinations(_set,2) #create all possible edges in each keyword set
            for edge in combinations:
                #does edge exist?
                if (self.has_edge(edge[0],edge[1])) | (self.has_edge(edge[1],edge[0])): #if it already has an edge like this
                    edge_count = self.edges[edge]['edge_count'] + 1 # increase the edge count by 1
                    self.edges[edge]['edge_count'] = edge_count # then assign the increased edge count
                    continue
                else:
                    self.add_edge(edge[0],edge[1],edge_count= 1,edge_type = 'keyword') #assign an edge count as 1

        for _column in self.attributes:
            self.add_graph_attribute(_column, self.attributes[_column])

    def draw_graph(self):
        edges = self.edges()
        weights = [self[u][v]['edge_count']**2 for u,v in edges]
        sizes = [2**self.nodes[u]['node_count'] * 100 for u in self.nodes]
        nx.draw(self,with_labels=True, width=weights, node_size=sizes)

    #get sub-graphs that are 1 or 2 nodes together
    def _get_subgraphs(self,max_size = 2):
        self.isolated_nodes = set()
        sub_graphs = ((len(g),g) for g in nx.connected_components(self))
        for size,nodes in sub_graphs:
            if size <= max_size: #then this graph is one we wantg
                self.isolated_nodes = self.isolated_nodes.union(nodes)

    #do nlp comparision with every other node and then join if threshold is >certain similarity
    def word_vector_compare(self, threshold=0.7,max_size=2):
        self.vector_d = dict()
        self._get_subgraphs(max_size=max_size)
        for node in self.nodes: #apply nlp to every node
            self.vector_d[node] = {'nlp': self.nlp_object(node)}
        #go through each isolated node and do vector similarity
        for node in self.isolated_nodes:
            for node2 in self.nodes:
                if node == node2:
                    continue
                similarity = self.vector_d[node]['nlp'].similarity(self.vector_d[node2]['nlp'])
                #add edge
                if(similarity > threshold):
                    self.add_edge(u_of_edge = node,v_of_edge = node2,similarity = similarity, edge_type = 'similarity', edge_count = 1)




# In[321]:


class TiddlyMap():

    def __init__(self, graph=None, text_attribute=None):
        self.graph = graph #KeywordGraph Object inherits from NetworkX Graph
        self.node_size_factor = 10 #for every increase in 1 node count, increase node size by this
        self.node_size_min = 5 #min node size
        self.edge_thickness_factor = 1
        self.edge_thickness_min = 1
        self.list_of_views = [] # list of JSON views
        self.tags = []
        self.quantiles = []
        self.edge_type_prefix_string = '$:/plugins/felixhayashi/tiddlymap/graph/edgeTypes/'
        self.node_type_prefix_string = '$:/plugins/felixhayashi/tiddlymap/graph/nodeTypes/'
        self.views_prefix_string = '$:/plugins/felixhayashi/tiddlymap/graph/views/'
        if text_attribute != None:
            self.assign_text_attribute(text_attribute)

    def assign_text_attribute(self, text_attribute):
        if text_attribute in self.graph.attributes:
            self.text_attribute = text_attribute
        else:
            print('Attribute does not exist in the graph')

    def add_networkx_graph(self,graph):
        self.graph = graph

    def append_networkx_graph(self,graph2):
        if self.graph == None: #there is no graph to append to
            self.add_networkx_graph(graph)
        else:
            self.graph = nx.compose(self.graph,graph2)

    def create_view(self,tags=['All'],edge_count_min=0,view_name = 'AllTagsAllEdges', similarity = False):

        #create nodes filter
        view_dict = dict()
        view_dict['text'] = ""
        view_dict['title'] = self.views_prefix_string + view_name + '/filter/nodes'

        tag_filters = []
        for tag in tags:
            tag_filters.append('[tag['+tag+']]')

        view_dict['filter'] = " ".join(tag_filters)
        self.list_of_views.append(view_dict)

        #create view title tiddler
        view_dict = dict()
        view_dict['title'] = self.views_prefix_string + view_name
        view_dict['config.physics_mode'] = 'true'
        view_dict['isview'] = 'true'
        view_dict['id'] = view_name
        view_dict["config.filter_nodes_by_edge_types"] = 'false'
        self.list_of_views.append(view_dict)

        #create edge view fitler
        view_dict = dict()
        view_dict['title'] = self.views_prefix_string + view_name + '/filter/edges'

        #loop through all edges
        edge_filters = ['-[prefix[_]]', '-[[tw-body:link]]', '-[[tw-list:tags]]', '-[[tw-list:list]]']
        for edge in self.list_of_edge_sizes_dicts:
            edge_size = self.get_edge_size(edge['title'])
            edge_name = edge['title'].split('/')[-1] # get the words after the last forward slash

            if((self.is_similarity_edge(edge['title'])) & (similarity == True)):  #if we want a similar edge then add it
                edge_filters.append('[['+edge_name+']]')
                continue
            if(edge_size >= edge_count_min) & (self.is_similarity_edge(edge['title'])==False): #we have a big enough keyword match
                edge_filters.append('[['+edge_name+']]')
            else:
                edge_filters.append('-[['+edge_name+']]') #minus sign means don't show

        view_dict['filter'] = " ".join(edge_filters)
        self.list_of_views.append(view_dict)

    def _test_for_edge_attributes(self,key,value):
        for u,v,data in self.graph.edges(data=True):
            if(data[key]==value):
                return True
        return False

    def _build_views(self): #build default views, with each tag different degress of connection
        #test to see if we have similarity edges
        if(self._test_for_edge_attributes(key='edge_type',value='similarity')):
            similarity_list = [True,False]
        else:
            similarity_list = [False]
        #edge_thresholds = np.linspace(start=self.edge_size_min, stop=self.edge_size_max, num=3)
        #edge_thresholds = {int(x) for x in edge_thresholds}#ensure only integers and no duplications
        for tag in self.tags:
            for edge_size in self.edge_sizes:
                for similarity_edge in similarity_list:
                    view_name = tag+' at least '+ str(edge_size) + ' connections ' + 'vector = ' + str(similarity_edge)
                    self.create_view(tags=[tag],edge_count_min=edge_size,view_name = view_name,similarity = similarity_edge)


    def build(self, node_count = 'node_count', edge_count = 'edge_count',percent_list = [0,0.25,0.5,0.75], tags = ['All','Top75%','Top50%','Top25%']):
        self.tags = tags
        self.percents = percent_list
        self.list_of_node_dicts = []
        self.list_of_node_sizes_dicts = []
        self.list_of_edge_sizes_dicts = []
        self.edge_titles = set() #list of all edge titles - distinct edge types
        node_sizes = set()
        all_node_sizes = np.array([node[1][node_count] for node in self.graph.nodes(data=True)])
        quantiles = np.quantile(all_node_sizes,percent_list) #so we can assign tags as to whether the node is in the top 25%, 50% etc. of nodes

        for node in self.graph.nodes:
            node_dict = dict()
            node_dict['title'] = node
            node_dict['tmap.id'] = node

            tag_list = []
            for i,quant in enumerate(quantiles):
                if self.graph.nodes[node][node_count] >= quant:
                    tag_list.append(tags[i])

            tag_string = " ".join(tag_list)
            node_dict['tags'] = tag_string

            #does node have edges
            edge_id_count = 0
            edge_dict = dict()
            for edge in self.graph.edges:
                if node == edge[0]: #only look in the left hand side of the edge tuple to avoid duplicating each edge
                    edge_id_string = node + "edge_" + str(edge_id_count)
                    edge_dict[edge_id_string] = {"to" : edge[1]}
                    edge_title = str(self.graph[edge[0]][edge[1]]['edge_type'])+"_edge_type_" + str(self.graph[edge[0]][edge[1]][edge_count])
                    edge_dict[edge_id_string]["type"] = edge_title #this needs to match the edge title
                    edge_id_count += 1 #each distinct edge needs a different ID
                    self.edge_titles.add(edge_title)

            node_dict['tmap.edges'] = str(edge_dict).replace("'",'\"')
            if self.text_attribute == None:
                node_dict['text'] = ""
            else:
                node_dict['text'] = "\n\n".join(self.graph.nodes[node][self.graph.attributes[self.text_attribute]])
            self.list_of_node_dicts.append(node_dict)
            #we need to go through each node and create a node of that size (from node_count)
            node_sizes.add(self.graph.nodes[node][node_count]) #add the node size to the set

        for node_size in node_sizes: #create the node tiddlers which define how big a node is and what keywords it includes
            node_size_dict = dict()
            node_title = self.node_type_prefix_string + 'node_type_' + str(node_size)
            node_size_dict['title'] = node_title

            node_style_dict = dict()
            #node_style_dict['shapeProperties'] = {'borderDashes' : 'False'}
            node_style_dict['font'] = {'size' : node_size*self.node_size_factor + self.node_size_min}

            node_size_dict['style'] = str(node_style_dict).replace("'",'\"')
            node_size_dict['priority'] = 1
            node_size_dict['scope'] = ''

            for node in self.graph.nodes:
                if self.graph.nodes[node][node_count] == node_size: #there is a match
                    scope_string = '[['+node+']]'
                    node_size_dict['scope'] += scope_string

            self.list_of_node_sizes_dicts.append(node_size_dict)

            #to do - go through and create notes with scope, also create edge types with thicknesses
        edge_sizes = set()
        for u,v,data in self.graph.edges(data=True):
            edge_sizes.add(data[edge_count]) #get a unique list of all the edge sizes

        #use this opportunity to set the max and min edge size
        self.edge_size_max = max(edge_sizes)
        self.edge_size_min = min(edge_sizes)
        self.edge_sizes = edge_sizes

        for edge_title in self.edge_titles: #this is a set of unique edge titles
            edge_size_dict = dict()
            edge_size_dict['title'] = self.edge_type_prefix_string + edge_title
            edge_size = self.get_edge_size(edge_title)
            edge_style_dict = dict()
            edge_style_dict['width'] =  edge_size * self.edge_thickness_factor + self.edge_thickness_min #way to scale up the edge thickness
            edge_style_dict['arrows'] = {'to' : {'enabled' : 'False'}}
            edge_size_dict['show-label'] = 'false'

            if self.is_similarity_edge(edge_title):
                #then we want red and dashes
                edge_style_dict['dashes'] = 'true'
                edge_style_dict['color'] = {"color":"rgba(206,38,43,1)"}

            edge_size_dict['style'] = str(edge_style_dict).replace("'",'\"')

            self.list_of_edge_sizes_dicts.append(edge_size_dict)

        self._build_views()

    def is_similarity_edge(self,edge_title):
        if(re.search('similarity',edge_title)):
            return True
        else:
            return False

    def get_edge_size(self,edge_title):
        edge_size = int(re.search('_(\d+)$',edge_title).group(1))
        return edge_size

    def write_tiddler_file(self, filename = 'tiddlers.json'):
        self.all_tiddlers = self.list_of_node_sizes_dicts + self.list_of_node_dicts + self.list_of_edge_sizes_dicts + self.list_of_views
        t=json.dumps(self.all_tiddlers) #tiddlywiki needs double quote escaped!!
        json_file = open(filename, "w+")
        json_file.truncate(0)
        json_file.write(t)
        json_file.close()



# In[209]:


def create_wordcloud(df, text_column, filename=None, **kwargs):
    """produces a wordcloud image
    text_column needs to be a string, if it isn't then transfer it before hand with list_to_string or set_to_string

    if filename is something, then it writes to a PNG file e.g. 'wordcloud.png'
    returns a wordcloud object"""
    wc = WordCloud(**kwargs)
    text = ' '.join(df[text_column])
    wc.generate_from_text(text)

    if filename:
        wc.to_file(filename)
    return wc

#@pysnooper.snoop()
def plot_wordcloud(df, text_column, row=None, **kwargs):
    """plot word clouds but separate the data along categorical variables of x and y

    x and y are columns in the dataframe"""
    def strip_axis_ticks_and_labels(ax):
        ax.tick_params(
                    axis='both',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,
                    left=False,# ticks along the top edge are off
                    labelbottom=False,
                    labelleft=False)

    if row:
        num_rows = len(df[row].unique())
        row_values = list(df[row].unique())
    else:
        num_rows = 1
        row_values = None


    #create matplotlib subplots
    fig,ax = plt.subplots(nrows=num_rows, ncols=1, figsize=(6,6))

    if row:
        #loop through the rows, filtering on teh dataframe by row value
         #create a wordcloud then put in axes
        for i,value in enumerate(row_values):
            df_temp = df[df[row] == value]

            wc_temp = create_wordcloud(df_temp, text_column, **kwargs)
            ax[i].imshow(wc_temp)
            strip_axis_ticks_and_labels(ax[i])
            ax[i].set_title(row + '=' + str(value))
    else:
        wc_temp = create_wordcloud(df, text_column, **kwargs)
        ax.imshow(wc_temp)
        strip_axis_ticks_and_labels(ax)

def add_to_set(original_set: set, new_item):
    """
    Takes a set of items and adds to it the new item. Designed to be used on a dataframe series
    A copy is made of the original set so that it is not altered, allowing the new set to be placed in a different column

    returns a new set"""
    assert type(original_set) == set


    copy_of_set = original_set.copy()
    if type(new_item) == list:
        new_item = set(new_item)
    if type(new_item) == set:
        copy_of_set = copy_of_set.union(new_item)
    else:
        copy_of_set.add(new_item)
    return copy_of_set

def string_to_list(my_string:str,**kwargs):
    return my_string.split(**kwargs)

def list_to_string(my_list:list, sep=" "):
    return sep.join(my_list)

def set_to_list(my_set:set):
    return list(my_set)

def set_to_string(my_set, sep=" "):
    return sep.join(my_set)



def create_clean_text(ser):
    if type(ser.iloc[0]) == set:
        #print('is set')
        ser = list(ser.fillna("").apply(lambda x: map(str,x)).str.join(" "))
        #print(ser[1:5])
        return ser
    if type(ser.iloc[0]) == list:
        #print('is list')
        return list(ser.fillna("").apply(lambda x: map(str,x)).str.join(" "))
    else:
        #print('else')
        return list(ser.fillna("").astype(str))

def create_set_from_series(ser):
    if type(ser.iloc[0]) == set:
        return ser.fillna("").apply(lambda x: map(str, x)).apply(set)
    elif type(ser.iloc[0]) == str:
        return ser.fillna("").str.split(" ").apply(set)
    elif type(ser.iloc[0]) == list:
        return ser.fillna("").str.join(" ").str.split(" ").apply(set)
    else:
        return ser.apply(lambda x: map(str,x))

def create_scattertext_plot(df, category_col:str, text_col:str, nlp, filename:str, label_match:str, label_name:str, label_other_name:str, metadata_col:str, **kwargs):
    """ creates a html file with an interactive scattertext plot

    Will delete an 'index' column if there is one as the scattertext function needs to create it
    label_match must be one of 2 entries in the category_col
    label_name is the user-friendly name given to a match, e.g. if label_match is 'Yes', you might want a more meaningful label such as 'A good week'
    label_other_name is the label for the other entry - e.g. 'A bad week'
    **kwargs goes into scattertext.produce_scattertext_explorer, e.g. minimum_term_frequency=8,

    :returns: nothing, but creates a HTML file"""
    if 'index' in df.columns:
        df.drop('index',axis=1,inplace=True)
    corpus = st.CorpusFromPandas(df,category_col=category_col,text_col=text_col, nlp=nlp).build()
    html = st.produce_scattertext_explorer(corpus,
                                      category=label_match,
                                      category_name=label_name,
                                      not_category_name=label_other_name,
                                      metadata=corpus.get_df()[metadata_col],
                                      save_svg_button=True,
                                           **kwargs
                                      )


    html_file = open(filename, 'wb')
    html_file.write(html.encode('utf-8'))
    html_file.close()

def nlp_pivot_vectors(df, nlp_column, prefix='nlp_vector_'):
    """ appends onto the original dataframe 300 columns, prefixed with 'prefix' argument that are the word vectors for
    whatever is in the NLP column (which must be an NLP object containing word vectors)
    Intended to be used in advance of UMAP-Learn dimensional reduction"""
    temp_dict = dict()
    for index, row in enumerate(df[nlp_column]):
        temp_dict[index] = row.vector.T.tolist() # transpose the vector so there will be one column per item in the vector
    temp_df = pd.DataFrame(temp_dict).T
    temp_df = temp_df.add_prefix(prefix)
    df = pd.concat([df,temp_df],axis=1)
    return df



def nlp_pivot_word_count(df, column_name, prefix='word_count_', **kwargs):
    """ Appends onto the original dataframe columns for every word contained in the column_name Series
    then a count of each word.This is  a wrapper for the sklearn CountVectorizer function.
    Text column (column_name) needs to be a string
    kwargs goes into the CountVectorizer function

    Intended to be used in advance of UMAP-Learn dimensional reduction"""
    cv = CountVectorizer(**kwargs)
    X = cv.fit_transform(df[column_name])
    names = cv.get_feature_names()
    temp_df = pd.DataFrame(data = X.toarray(), columns = names)
    temp_df = temp_df.add_prefix(prefix)
    df = pd.concat([df,temp_df],axis=1)
    return df

def create_umap_plot(df, text_column:str, other_columns:list, filename:str, hover_columns:list, label:str, label_categorical=True, text_processing = 'vector', n_neighbours = 2, metric = None, y=None, **kwargs):
    """
    dimensional reduction on text fields - a few options, can use vector similarities or word_counts, and add additional columns
    pre-processing must be done on other columns to turn them into numerical / boolean values first

    text_column needs to be a string where words are separated by spaces
    If vector similarites are used then the distance metric used is cosine, if word counts, then hellinger is used (in umap.UMAP function)
    If using word vectors then the text_column needs to contain NLP objects with vector embeddings.

    hover_columns: data to show in any mouse hover-over
    label: what column to use for colouring the plot
    label_categorical: is the label a categorical column or a continuous column (will impact the colouring scheme)
    n_neighbours - fed into umap.UMAP() funciton for nearest neighbours. For short comments, low numbers give the best clusters
    metric: can be used to override the distance metric of cosine or hellinger (str)
    y: column to train the UMAP fit function (optional)
    text_processing: one of 'vector', 'count', or None in which case text_column is ignored and just other_columns used in projection
    **kwargs goes into umap.plot.interactive()

    returns: nothing, but saves a html file"""

    #set the umap reducer metric
    if metric:
        metric = metric # the user has chosen to override
    elif text_processing == 'vector':
        metric = 'cosine'
    elif text_processing == 'count':
        metric = 'hellinger'
    else:
        metric = 'euclidean' #if no text processing, and no metric set

    reducer = umap.UMAP(n_neighbors=n_neighbours, metric=metric)

    #create the embedding dataframe

    if text_processing == 'vector':
        df_word_embeddings = nlp_pivot_vectors(df, text_column).filter(regex='nlp_vector_*')
    elif text_processing == 'count':
        df_word_embeddings = nlp_pivot_word_count(df, text_column).filter(regex='word_count_*')
    else:
        df_word_embeddings = pd.DataFrame()

    df_embedding = pd.concat([df[other_columns], df_word_embeddings],axis=1)

    #fit the reducer to the embedding dataframe
    if y:
        y=df[y]

    umap_embedding = reducer.fit(df_embedding,y=y)

    #work out the label or value parameter for the umap.plot.interactive function
    #if the column is categorical it will be a label, if not a value

    if label_categorical == True:
        p = umap.plot.interactive(
            umap_embedding,
            labels = df[label],
            hover_data=df[hover_columns],
            **kwargs)
    else:
        #it is a continuous colour-map
        p = umap.plot.interactive(
            umap_embedding,
            values = df[label],
            hover_data=df[hover_columns],
            **kwargs)

    output_file(filename)
    save(p)



# ## Importing a spreadsheet of comments
# We'll have a really simple spreadsheet of only 7 comments

# ## Pre-processing
# First we apply a natural language model onto the comment columns, using our nlp object we defined at the beginning.
#
# Note the defaults in the function call tell it to produce another dataframe column called 'nlp'

# In[ ]:
