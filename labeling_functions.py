from cgitb import text
from multiprocessing.sharedctypes import Value
from nis import match
import re
import time

from nltk import ngrams
from nltk.tokenize import WhitespaceTokenizer, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
# nltk.download('stopwords')
import gensim
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import spacy

#loading the english language small model of spacy
en = spacy.load('en_core_web_sm')

pico2labelMap = { 'P' : 1, 'I' : 1, 'O' : 1, 'S': 1, '-P' : -1, '-I' : -1, '-O' : -1, '-S' : -1, '!P' : 0, '!I' : 0, '!O' : 0, '!S' : 0 }


def loadStopWords():

    stopwords_lf = []

    # NLTK
    nltk_stopwords = list(stopwords.words('english'))
    # print( 'Total number of stopwords in NLTK: ', len( nltk_stopwords ) )
    stopwords_lf.extend( nltk_stopwords )

    # gensim
    # print( 'Total number of stopwords in Gensim: ', len( STOPWORDS ) )
    stopwords_lf.extend( STOPWORDS )

    # scikit learn
    # print( 'Total number of stopwords in scikit learn: ', len( ENGLISH_STOP_WORDS ) )
    stopwords_lf.extend( ENGLISH_STOP_WORDS )

    # spacy
    spacy_stopwords = en.Defaults.stop_words
    # print( 'Total number of stopwords in Spacy: ', len( spacy_stopwords ) )
    stopwords_lf.extend( spacy_stopwords )

    # additional stopwords
    additional_stopwords = ['of']
    stopwords_lf.extend(additional_stopwords)

    return list( set(stopwords_lf) )


def char_to_word_index(ci, sequence, tokens):
    """
    Given a character-level index (offset),
    return the index of the **word this char is in**
    """
    i = None
    for i, co in enumerate(tokens):
        if ci == co:
            return i
        elif ci < co:
            return i - 1
    return i


def get_word_index_span(char_offsets, sequence, tokens):
    char_start, char_end = char_offsets
    return (char_to_word_index(char_start, sequence, tokens),
            char_to_word_index(char_end, sequence, tokens))


def spansToLabels(matches, df_data, picos:str):

    df_data_labels = []
    #Spans to labels
    for counter, match in enumerate(matches):

        abstain_lab = '!'+picos
        L = dict.fromkeys(range( len(list(df_data['offsets'])[counter]) ), abstain_lab) # initialize with abstain labels
        numerical_umls_labels = dict()

        for (char_start, char_end), term, lab in match:
            
            # None labels are treated as abstains
            if not lab:
                continue

            start, end = get_word_index_span(
                (char_start, char_end - 1), list(df_data['text'])[counter], list(df_data['offsets'])[counter]
            )

            for i in range(start, end + 1):
                L[i] = lab
        
        # Fetch numerical labels
        for k,v in L.items():
            numerical_umls_labels[k] = pico2labelMap[ v ]
        
        df_data_labels.append( numerical_umls_labels )

    return df_data_labels


def get_label_dict(ont_list:list, picos: str) -> dict:

    ont_dict = dict()

    for l in ont_list:
        if l not in ont_dict and len(l) > 1:
            ont_dict[ l ] = picos
        
    return ont_dict


def match_term(term, dictionary, case_sensitive, lemmatize=True):
    """
    Parameters
    ----------
    term
    dictionary
    case_sensitive
    lemmatize   Including lemmas improves performance slightly
    Returns
    -------
    """
    if (not case_sensitive and term.lower() in dictionary) or term in dictionary:
        label_produced = None
        if (not case_sensitive and term.lower() in dictionary):
            label_produced = dictionary[ term.lower() ]
        if term in dictionary:
            label_produced = dictionary[ term ]
        return [True, label_produced]

    if (case_sensitive and lemmatize) and term.rstrip('s').lower() in dictionary:
        label_produced = dictionary[ term.rstrip('s').lower() ]
        return [True, label_produced]

    elif (not case_sensitive and lemmatize) and term.rstrip('s') in dictionary:
        label_produced = dictionary[ term.rstrip('s') ]
        return [True, label_produced]

    return [False, None]


def OntologyLabelingFunctionX(corpus_text_series, 
                              corpus_words_series,
                              corpus_offsets_series,
                              source_terms: dict,
                              picos: str,
                              fuzzy_match: bool,
                              stopwords_general = list,
                              max_ngram: int = 5,
                              case_sensitive: bool = False,
                              longest_match_only = True):

    # get label dict
    source_terms = get_label_dict( source_terms, picos='I' )

    # Add bigrams in case 
    if fuzzy_match == True:
        pass
        # source_bigrams = LFutils.build_word_graph( source_terms, picos )
        # source_terms.update(source_bigrams)

    # Add stopwords to the dictionary if 
    if stopwords_general:
        for sw in stopwords_general:
            sw_negative = '-'+picos
            if sw not in source_terms:
                source_terms[sw] = sw_negative


    corpus_matches = []
    longest_matches = []

    for words_series, offsets_series, texts_series in zip(corpus_words_series, corpus_offsets_series, corpus_text_series):

        matches = []

        for i in range(0, len(words_series)):

            match = None
            start = offsets_series[i]

            for j in range(i + 1, min(i + max_ngram + 1, len(words_series) + 1)):
                end = offsets_series[j - 1] + len(words_series[j - 1])

                # term types: normalize whitespace & tokenized + whitespace
                for term in [
                    re.sub(r'''\s{2,}''', ' ', texts_series[start:end]).strip(),
                    ' '.join([w for w in words_series[i:j] if w.strip()] )
                ]:
                    match_result = match_term(term, source_terms, case_sensitive)
                    if match_result[0] == True:
                        match = end
                        break

                if match:
                    term = re.sub(r'''\s{2,}''', ' ', texts_series[start:match]).strip()
                    matches.append(( [start, match], term, match_result[-1] ))

        corpus_matches.append( matches )

        if longest_match_only:
            # sort on length then end char
            matches = sorted(matches, key=lambda x: x[0][-1], reverse=1)
            f_matches = []
            curr = None
            for m in matches:
                if curr is None:
                    curr = m
                    continue
                (i, j), _, _ = m
                if (i >= curr[0][0] and i <= curr[0][1]) and (j >= curr[0][0] and j <= curr[0][1]):
                    pass
                else:
                    f_matches.append(curr)
                    curr = m
            if curr:
                f_matches.append(curr)

            # if f_matches:
            longest_matches.append( f_matches )

    if longest_match_only:
        assert len( longest_matches ) == len(corpus_words_series)
        return longest_matches
    else:
        assert len( corpus_matches ) == len(corpus_words_series)
        return corpus_matches