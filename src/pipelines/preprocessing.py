import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
import re
import string
import spacy
from spacy.matcher import Matcher
import configparser


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


dale_chall_df=pd.read_csv("./src/pipelines/Dale_Chall_List.txt", header=None, delimiter="\t")
dale_chall = dale_chall_df.iloc[:,0].tolist()

def get_word_list(para):
    word_list = re.sub('['+string.punctuation+']', '', para).split()
    return(word_list)

def calc_avg_sent_len(para_series):
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
        #print(number_of_sentences,)
        len_sent.append(avg_len)
    return(len_sent)

def count_syllables(word):
    vowel_runs = len(VOWEL_RUNS.findall(word))
    exceptions = len(EXCEPTIONS.findall(word))
    additional = len(ADDITIONAL.findall(word))
    #print(word, max(1, vowel_runs - exceptions + additional))
    return max(1, vowel_runs - exceptions + additional)

def cal_hard_words(para_series):
    ptage_hard = []
    more_2 =[]
    less_eq_2 =[]
    for para in para_series:
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
    ptage_per_pro = []
    for para in para_series:
        word_list1 = get_word_list(para)
        total_per_pro = sum(el in word_list1 for el in persona_pronouns)
        total_words = len(word_list1)
        ptage_personal_pronouns = total_per_pro/total_words        
        ptage_per_pro.append(ptage_personal_pronouns)
    return(ptage_per_pro)


def is_passive(sentence):
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
    from nltk.tokenize import sent_tokenize
    ptage_passive = []
    for para in para_series:
        passive_count=0
        sentences = sent_tokenize(para)
        for sent in sentences:
            if is_passive(sent)=="Passive":
                passive_count=+1
        passives=passive_count/len(sentences)
        ptage_passive.append(passives)
    return(ptage_passive)

def cal_per_unique(para_series):
    ptage_unique = []
    for para in para_series:
        word_list1 = get_word_list(para)
        unique=set(word_list1)
        per_unique = len(unique)/len(word_list1)
        ptage_unique.append(per_unique)
    return(ptage_unique)

def get_pps(doc):
    "Function to get PPs from a parsed document."
    pps = []
    for token in doc:
        # Try this with other parts of speech for different subtrees.
        if token.pos_ == 'ADP':
            pp = ' '.join([tok.orth_ for tok in token.subtree])
            pps.append(pp)
    return pps

def cal_per_pps(para_series):
    ptage_pps = []
    from nltk.tokenize import sent_tokenize
    for para in para_series:
        pps_count=0
        sentences = sent_tokenize(para)
        for sent in sentences:
            doc = nlp(sent)
            count_pps_sent = len(get_pps(doc))
            pps_count=+count_pps_sent
        
        pps=pps_count/len(sentences)
        ptage_pps.append(pps)
    return(ptage_pps)

def cal_per_non_dc(para_series):
    ptage_non_dc = []
    for para in para_series:
        word_list1 = get_word_list(para)
        non_dc = list(set(word_list1)-set(dale_chall))
        p_n_dc = len(non_dc)/len(word_list1)
        ptage_non_dc.append(p_n_dc)
    return(ptage_non_dc)


def preprocess(prep_train_df, excerpt):

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

    ptage_pps= cal_per_pps(prep_train_df[excerpt])
    prep_train_df['ptage_pps'] = pd.Series(ptage_pps).values

    ptage_non_dc= cal_per_non_dc(prep_train_df[excerpt])
    prep_train_df['ptage_non_dale_chall'] = pd.Series(ptage_non_dc).values

    return prep_train_df