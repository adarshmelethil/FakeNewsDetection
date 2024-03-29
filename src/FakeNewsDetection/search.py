import json
from gensim.models.phrases import Phrases, Phraser
from gensim import similarities
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import spacy
from .raw_data import docs

def get_sents(text):
    return [tokenize(sent) for sent in list(text.sents)]

def tokenize(text):
    tokens = []
    for word in text:
        tokens.append(word)
    return tokens

def preprocess(segments, dct=None, bigram=None):
    processed_segments = []
    for seg in segments:
        processed_seg = []
        for word in seg:
            if True in [word.is_space, word.is_stop, word.is_punct]: 
                continue
            word = word.lemma_
            word = word.lower()
            processed_seg.append(word)
        processed_segments.append(processed_seg)

    if bigram is None:
        phrases = Phrases(processed_segments, min_count=3, threshold=3) 
        bigram = Phraser(phrases)

    processed_segments = bigram[processed_segments]

    if dct is None:
        dct = Dictionary(processed_segments)
    else:
        dct.add_documents(processed_segments)

    return [dct.doc2bow(line) for line in processed_segments], dct, processed_segments, bigram

def preprocess_query(query, nlp, dct, bigram):
    query = nlp(query)
    query = tokenize(text=query)
    query, dct, processed_segments, bigram = preprocess(segments=[query], dct=dct, bigram=bigram)
    query = query[0]
    return query

def preprocess_text(text, nlp, dct=None):
    text = nlp(text)
    segments = get_sents(text=text)
    bag_of_words, dct, processed_segments, bigram = preprocess(segments=segments, dct=dct)
    return segments, dct, bag_of_words, processed_segments, bigram

def get_original(segments):
    return [' '.join([word.text.strip() for word in seg]) for seg in segments]

def tfidf_search(query, segments, segments_original, dct, bag_of_words, nlp):
    tfidf = TfidfModel(bag_of_words)
    index = similarities.SparseMatrixSimilarity(tfidf[bag_of_words], num_features=len(dct.dfs))
    results = list(enumerate(index[tfidf[query]]))
    results = [dict(
            original=segments_original[res[0]],
            article_id=float(res[0]),
            relevance=float(res[1]),
        ) 
        for res in results]
    results.sort(key=lambda res: res['relevance'], reverse = True) 
    return results

def search(query, text, nlp, threshold):
    segments, dct, bag_of_words, processed_segments, bigram = preprocess_text(text=text, nlp=nlp)
    segments_original = get_original(segments=segments)
    query = preprocess_query(query=query, nlp=nlp, dct=dct, bigram=bigram)
    
    results = tfidf_search(
                query=query,
                segments=segments,
                segments_original=segments_original,
                dct=dct,
                bag_of_words=bag_of_words,
                nlp=nlp
            )

    return [ res for res in results if res['relevance'] > threshold]

def search_multiple_docs(query, docs, nlp, threshold):
    text = '\n\n'.join(docs)
    return search(query, text=text, nlp=nlp, threshold=threshold)


def print_results(results):
    for res in results:
        print(json.dumps(res,indent=4))

def main():
    nlp = spacy.load("en_core_web_sm")
    query = '''human rights watch says that women are not denined justice'''
    results = search_multiple_docs(query=query, docs=docs, nlp=nlp, threshold=0.1)
    print_results(results)

if __name__ == '__main__':
    main()
