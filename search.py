import json
import gensim.downloader as api
from gensim import similarities
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import spacy
import copy

def split_modes(mode=None):
    available_modes = dict(
        sent="SENT",
        no_split="NO_SPLIT",
    )
    return available_modes[mode]

def get_sents(text):
    return [tokenize(sent) for sent in list(text.sents)]

def tokenize(text):
    tokens = []
    for word in text:
        tokens.append(word)
    return tokens

def preprocess(segments, dct=None):
    processed_data = []
    for seg in segments:
        processed_seg = []
        for word in seg:
            if True in [word.is_space, word.is_stop, word.is_punct]: 
                continue
            word = word.lemma_
            word = word.lower()
            processed_seg.append(word)
        processed_data.append(processed_seg)
    if dct is None:
        dct = Dictionary(processed_data)
    else:
        dct.add_documents(processed_data)
    return [dct.doc2bow(line) for line in processed_data], dct

def create_index( bag_of_words, dct, nlp):
    tfidf = TfidfModel(bag_of_words)
    index = similarities.SparseMatrixSimilarity(tfidf[bag_of_words], num_features=len(dct.dfs))
    return index, tfidf

def eval_tfidf(query, index, tfidf, nlp):
    result = index[tfidf[query]]
    return list(enumerate(result))

def get_original(segments):
    return [' '.join([word.text for word in seg]) for seg in segments]

def search(query, segments, nlp, bag_of_words, dct,threshold):
    # text = nlp(text)

    segments_original = get_original(segments=segments)
    index, tfidf = create_index(dct=dct, bag_of_words=bag_of_words, nlp=nlp)
    results = eval_tfidf(query=query, index=index, tfidf=tfidf, nlp=nlp)

    results = [dict(
            original=segments_original[res[0]],
            # processed=list(bag_of_words),
            relevance=float(res[1]),
        ) 
        for res in results if res[1] > threshold]
    results.sort(key=lambda res: res['relevance'], reverse = True) 
    return results

def preprocess_query(query, nlp, dct):
    query = nlp(query)
    query = tokenize(text=query)
    query, dct = preprocess(segments=[query], dct=dct)
    query = query[0]
    return query

def preprocess_text(text, nlp, dct=None):
    text = nlp(text)
    segments = get_sents(text=text)
    bag_of_words, dct = preprocess(segments=segments, dct=dct)
    return segments, dct, bag_of_words

def search_documents(query, docs, nlp, threshold=0):
    dct = None
    docs_segments = []
    for text in docs:
        segments, dct, bag_of_words = preprocess_text(text=text, nlp=nlp, dct=dct)
        docs_segments.append(segments)

    all_results = []
    query = preprocess_query(query=query, nlp=nlp, dct=dct)

    for segments in docs_segments:
        results = search(query=query,
            segments=segments, 
            nlp=nlp, 
            dct=dct, 
            bag_of_words=bag_of_words, 
            threshold=threshold)
        all_results.append(results)
    return all_results

def print_all_results(all_results):
    for results in all_results:
        print("================= new doc")
        for res in results:
            print(json.dumps(res,indent=4))

def main():
    nlp = spacy.load("en_core_web_sm")

    doc1 = '''
South African doctor invents female condoms with 'teeth' to fight rape
(CNN) -- South African Dr. Sonnet Ehlers was on call one night four decades ago when a devastated rape victim walked in. Her eyes were lifeless; she was like a breathing corpse.

"She looked at me and said, 'If only I had teeth down there,'" recalled Ehlers, who was a 20-year-old medical researcher at the time. "I promised her I'd do something to help people like her one day."

Forty years later, Rape-aXe was born.

Ehlers is distributing the female condoms in the various South African cities where the World Cup soccer games are taking place.

The woman inserts the latex condom like a tampon. Jagged rows of teeth-like hooks line its inside and attach on a man's penis during penetration, Ehlers said.

Once it lodges, only a doctor can remove it -- a procedure Ehlers hopes will be done with authorities on standby to make an arrest.'''

    doc2 = '''
"It hurts, he cannot pee and walk when it's on," she said. "If he tries to remove it, it will clasp even tighter... however, it doesn't break the skin, and there's no danger of fluid exposure."

Ehlers said she sold her house and car to launch the project, and she planned to distribute 30,000 free devices under supervision during the World Cup period.

"I consulted engineers, gynecologists and psychologists to help in the design and make sure it was safe," she said.

After the trial period, they'll be available for about $2 a piece. She hopes the women will report back to her.

"The ideal situation would be for a woman to wear this when she's going out on some kind of blind date ... or to an area she's not comfortable with," she said.

The mother of two daughters said she visited prisons and talked to convicted rapists to find out whether such a device would have made them rethink their actions.

Some said it would have, Ehlers said.
    '''
    doc3 = '''
Critics say the female condom is not a long-term solution and makes women vulnerable to more violence from men trapped by the device.

It's also a form of "enslavement," said Victoria Kajja, a fellow for the Centers for Disease Control and Prevention in the east African country of Uganda. "The fears surrounding the victim, the act of wearing the condom in anticipati
on of being assaulted all represent enslavement that no woman should be subjected to."

Kajja said the device constantly reminds women of their vulnerability.

"It not only presents the victim with a false sense of security, but psychological trauma," she added. "It also does not help with the psychological problems that manifest after assaults."

However, its one advantage is it allows justice to be served, she said.

Various rights organizations that work in South Africa declined to comment, including Human Rights Watch and Care International.
    '''
    doc4 = '''
South Africa has one of the highest rape rates in the world, Human Rights Watch says on its website. A 2009 report by the nation's Medical Research Council found that 28 percent of men surveyed had raped a woman or girl, with one in
20 saying they had raped in the past year, according to Human Rights Watch.

In most African countries, rape convictions are not common. Affected women don't get immediate access to medical care, and DNA tests to provide evidence are unaffordable.

"Women and girls who experience these violations are denied justice, factors that contribute to the normalization of rape and violence in South African society," Human Rights Watch says.

Women take drastic measures to prevent rape in South Africa, Ehlers said, with some wearing extra tight biker shorts and others inserting razor blades wrapped in sponges in their private parts.

Critics have accused her of developing a medieval device to fight rape.

"Yes, my device may be a medieval, but it's for a medieval deed that has been around for decades," she said. "I believe something's got to be done ... and this will make some men rethink before they assault a woman."
107947.txt (END)
    '''
    docs = [doc1, doc2, doc3, doc4]
    query = '''Women take drastic measures to prevent rape'''
    # segments, dct, bag_of_words = preprocess_text(text=text, nlp=nlp)
    # query = preprocess_query(query=query, nlp=nlp, dct=dct)
    # results = search(query=query, segments=segments, nlp=nlp, dct=dct, bag_of_words=bag_of_words, threshold=0.0)
    # print(results)

    all_results = search_documents(query=query, docs=docs, nlp=nlp, threshold=0)
    print_all_results(all_results=all_results)
main()
