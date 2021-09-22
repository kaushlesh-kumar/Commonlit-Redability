import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
import re
import string
import spacy
from spacy.matcher import Matcher
import configparser
# import smogn

# Some regex for calulating the number of syllables - source - https://datascience.stackexchange.com/a/89312
VOWEL_RUNS = re.compile("[aeiouy]+", flags=re.I)
EXCEPTIONS = re.compile(
    # fixes trailing e issues:
    # smite, scared
    "[^aeiou]e[sd]?$|"
    # fixes adverbs:
    # nicely
    + "[^e]ely$",
    flags=re.I
)
ADDITIONAL = re.compile(
    # fixes incorrect subtractions from exceptions:
    # smile, scarred, raises, fated
    "[^aeioulr][lr]e[sd]?$|[csgz]es$|[td]ed$|"
    # fixes miscellaneous issues:
    # flying, piano, video, prism, fire, evaluate
    + ".y[aeiou]|ia(?!n$)|eo|ism$|[^aeiou]ire$|[^gq]ua",
    flags=re.I
)

persona_pronouns = ['I','he','him','her','it','me','she','them','they','us','we','you','He','Him','Her','It','Me','She','Them','They','Us','We','You']

try:
    nlp = spacy.load('en_core_web_sm')
except:
    nlp = spacy.load('en')
matcher = Matcher(nlp.vocab)

# Loading the Dale-Chall list - https://en.wikipedia.org/wiki/Dale%E2%80%93Chall_readability_formula
dale_chall_df=pd.read_csv("Dale_Chall_List.txt", header=None, delimiter="\t")
dale_chall = dale_chall_df.iloc[:,0].tolist()

def get_word_list(para):
    """This tokenizes the words in a given paragraph

    Args:
        para (String): The block of text (passage) for processing
    """
    word_list = re.sub('['+string.punctuation+']', '', para).split()
    return(word_list)

def calc_avg_sent_len(para_series):
    """This calulates the average length of sentences in the passage. 
    Hypothesis - Longer sentences reduce readability 

    Args:
        para_series (pandas.series): The text column from the dataframe
    """
    len_sent = []
    from nltk.tokenize import sent_tokenize
    for para in para_series:
        number_of_sentences = len(sent_tokenize(para))
        num_word_all_sent =[]
        for sent in sent_tokenize(para):
            num_words = len(get_word_list(sent))
            num_word_all_sent.append(num_words)
        total_length = sum(num_word_all_sent)
        avg_len = total_length/number_of_sentences
        len_sent.append(avg_len)
    return(len_sent)

def count_syllables(word):
    """This calculates the number of syllables in a given word

    Args:
        word (String): The word in which syllables need to be counted

    Returns:
        int: Number of syllables
    """
    vowel_runs = len(VOWEL_RUNS.findall(word))
    exceptions = len(EXCEPTIONS.findall(word))
    additional = len(ADDITIONAL.findall(word))
    #print(word, max(1, vowel_runs - exceptions + additional))
    return max(1, vowel_runs - exceptions + additional)

def cal_hard_words(para_series):
    """This calculates the percentage of hard (more than 2 syllable words)in the passage
    Hypothesis - Higher proportion of hard words reduce readability 

    Args:
        para_series (pandas.series): The text column from the dataframe
    """
    ptage_hard = []
    for para in para_series:
        more_2 =[]
        less_eq_2 =[]
        word_list = get_word_list(para)
        for word in word_list:
            count=count_syllables(word)
            if count > 2:
                more_2.append(count)
            else:
                less_eq_2.append(count)
                
        ptage_hard.append(len(more_2)/(len(more_2)+len(less_eq_2))*100)
    return(ptage_hard)

def cal_per_personal_pronouns(para_series):
    """This calculates the percentage of personal pronouns in the passage
    Hypothesis - Higher proportion of personal pronouns improve readability

    Args:
        para_series (pandas.series): The text column from the dataframe
    """
    ptage_per_pro = []
    for para in para_series:
        word_list1 = get_word_list(para)
        total_per_pro = sum(el in word_list1 for el in persona_pronouns)
        total_words = len(word_list1)
        ptage_personal_pronouns = total_per_pro/total_words        
        ptage_per_pro.append(ptage_personal_pronouns)
    return(ptage_per_pro)


def is_passive(sentence):
    """This checks if the sentence is in passive voice

    Args:
        sentence (string): Sentence tokens from the given passage

    Returns:
        String: If the Sentence is in Active or Passive voice
    """
    doc = nlp(sentence)
    passive_rule = [{'DEP': 'nsubjpass'}, {'DEP': 'aux', 'OP': '*'}, {'DEP': 'auxpass'}, {'TAG': 'VBN'}]
    matcher.add('Passive', [passive_rule])
    matches = matcher(doc)
    #print(sentence, matches)
    if matches:
        return "Passive"
    else:
        return "Active"

def cal_per_passive(para_series):
    """This calculates the percentage of passive voice sentences in the passage
    Hypothesis - Higher proportion of passive sentences reduce readability

    Args:
        para_series (pandas.series): The text column from the dataframe
    """
    from nltk.tokenize import sent_tokenize
    ptage_passive = []
    for para in para_series:
        passive_count=0
        sentences = sent_tokenize(para)
        for sent in sentences:
            if is_passive(sent)=="Passive":
                passive_count = passive_count+1
        passives=passive_count/len(sentences)
        ptage_passive.append(passives)
    return(ptage_passive)

def cal_per_unique(para_series):
    """This calculates the percentage of unique words in the passage
    Hypothesis - Higher proportion of unique words reduce readability

    Args:
        para_series (pandas.series): The text column from the dataframe
    """
    ptage_unique = []
    for para in para_series:
        word_list1 = get_word_list(para)
        unique=set(word_list1)
        per_unique = len(unique)/len(word_list1)
        ptage_unique.append(per_unique)
    return(ptage_unique)

def cal_no_unique(para_series):
    """This calculates the number of unique words in the passage
    Hypothesis - Higher number of unique words reduce readability

    Args:
        para_series (pandas.series): The text column from the dataframe
    """
    no_unique = []
    for para in para_series:
        word_list1 = get_word_list(para)
        unique=set(word_list1)
        #per_unique = len(unique)/len(word_list1)
        no_unique.append(len(unique))
    return(no_unique)

def get_pps(doc):
    """This identifies prepositional phrases from a passage

    Args:
        doc (string): The sentence

    Returns:
        List: List of prepositional phrases idetified
    """
    "Function to get PPs from a parsed document."
    pps = []
    for token in doc:
        # Try this with other parts of speech for different subtrees.
        if token.pos_ == 'ADP':
            pp = ' '.join([tok.orth_ for tok in token.subtree])
            pps.append(pp)
    return pps

def cal_per_pps(para_series):
    """This calualates the percentage of prepositional phrases in the passage
    Hypothesis - Higher proportion of prepositional phrases in the passage reduce the redability

    Args:
        para_series (pandas.series): The text column of the dataframe
    """
    ptage_pps = []
    from nltk.tokenize import sent_tokenize
    for para in para_series:
        pps_count=0
        sentences = sent_tokenize(para)
        for sent in sentences:
            doc = nlp(sent)
            count_pps_sent = len(get_pps(doc))
            pps_count= pps_count + count_pps_sent
        
        pps=pps_count/len(sentences)
        ptage_pps.append(pps)
    return(ptage_pps)

def cal_per_non_dc(para_series):
    """This calculates the percentage of words in the passage which are not present in the Dale - Chall list
    Hypothesis - Higher proportion of words not present in the list (4th grade vocabulary) reduce the redability

    Args:
        para_series (pandas.series): The text column of the dataframe
    """
    ptage_non_dc = []
    for para in para_series:
        word_list1 = get_word_list(para)
        non_dc = list(set(word_list1)-set(dale_chall))
        p_n_dc = len(non_dc)/len(word_list1)
        ptage_non_dc.append(p_n_dc)
    return(ptage_non_dc)

def preprocess(prep_train_df, excerpt):
    """The main function which calls all the other function to create the features

    Args:
        prep_train_df (dataframe): The dataframe containing the text to be analyzed
        excerpt (String): The name of the column containing the text

    Returns:
        dataframe: The dataframe with the calculated features appended to it
    """

    len_sent = calc_avg_sent_len(prep_train_df[excerpt])
    prep_train_df['avg_sent_len'] = pd.Series(len_sent).values

    ptage_hard = cal_hard_words(prep_train_df[excerpt])
    prep_train_df['ptage_hard'] = pd.Series(ptage_hard).values

    ptage_per_pro = cal_per_personal_pronouns(prep_train_df[excerpt])
    prep_train_df['ptage_personal_pronouns'] = pd.Series(ptage_per_pro).values

    ptage_passive= cal_per_passive(prep_train_df[excerpt])
    prep_train_df['ptage_passive_sentences'] = pd.Series(ptage_passive).values

    ptage_unique= cal_per_unique(prep_train_df[excerpt])
    prep_train_df['ptage_unique_words'] = pd.Series(ptage_unique).values

    no_unique= cal_no_unique(prep_train_df[excerpt])
    prep_train_df['no_unique_words'] = pd.Series(no_unique).values

    ptage_pps= cal_per_pps(prep_train_df[excerpt])
    prep_train_df['ptage_pps'] = pd.Series(ptage_pps).values

    ptage_non_dc= cal_per_non_dc(prep_train_df[excerpt])
    prep_train_df['ptage_non_dale_chall'] = pd.Series(ptage_non_dc).values

    # prep_train_df= balancing_dataset(prep_train_df)

    return prep_train_df