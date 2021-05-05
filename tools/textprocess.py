import numpy as np
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pke
import itertools
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords as Stopwords


class TextProcess(object):
    def __init__(
        self,
    ):
        self.spacy_model = spacy.load("en_core_web_sm")
        self.sent_analyzer = SentimentIntensityAnalyzer()
        self.text = None
        self.text_file = None

        self.chunks_emb = None
        self.ents_text_emb = None
        self.keyphrases_emb = None

        self.model = SentenceTransformer("distilbert-base-nli-mean-tokens")

    def load_text(self, text):
        self.text = text
        self.sp_text = self.spacy_model(text)

        if not self.text_file:
            open("temp_123.txt", "w", encoding="utf-8").write(self.text)
            self.text_file = "temp_123.txt"

    def load_text_file(self, text_file):
        self.text_file = text_file
        text = open(self.text_file, "r", encoding="utf-8").read()
        self.load_text(text)

    def get_entities(self):
        self.ents = []
        self.ents_text = []
        for each in self.sp_text.ents:
            self.ents.append([each.text, each.start_char, each.end_char, each.label_])
            self.ents_text.append(each.text)

    def get_keyphrases(self, thershold=0.02):
        keyphrases = []
        self.keyphrases = {}
        for extractor in [
            pke.unsupervised.TopicRank(),
            pke.unsupervised.YAKE(),
            pke.unsupervised.PositionRank(),
        ]:
            extractor.load_document(input=self.text_file, language="en")
            extractor.candidate_selection()
            extractor.candidate_weighting()
            keyphrases.extend(extractor.get_n_best(n=10))
        for each in keyphrases:
            try:
                self.keyphrases[each[0]] += each[1]
                self.keyphrases[each[0]] /= 2
            except Exception:
                self.keyphrases[each[0]] = each[1]
        self.keyphrases = sorted(
            self.keyphrases.items(), key=lambda x: x[1], reverse=True
        )
        self.filtered_keyphrases = []
        for each in self.keyphrases:
            if each[1] > thershold:
                self.filtered_keyphrases.append(each[0])

        return self.keyphrases

    def get_sentiment(self):
        sentiment = 0.0
        sentiments = []
        self.sentiment = {}
        for sentence in self.text.splitlines():
            score = self.sent_analyzer.polarity_scores(sentence)
            if score["compound"] >= 0.05:
                sentiments.append(1)
            elif score["compound"] <= -0.05:
                sentiments.append(-1)
            else:
                sentiments.append(0)

            sentiment += score["compound"]

        self.sentiment["compound"] = sentiment / len(self.text.splitlines())
        self.sentiment["list"] = sentiments

    def get_chunks(self):

        text = self.text
        sentence_re = r"""(?:(?:[A-Z])(?:.[A-Z])+.?)|(?:\w+(?:-\w+)*)|
        (?:\$?\d+(?:.\d+)?%?)|(?:...|)(?:[][.,;"\'?():-_`])"""
        lemmatizer = nltk.WordNetLemmatizer()
        # stemmer = nltk.stem.porter.PorterStemmer()
        grammar = r"""
            NBAR:
                # Nouns and Adjectives,terminated with Nouns
                {<NN.*|JJ>*<NN.*>}
            NP:
                {<NBAR>}
                {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
        """
        chunker = nltk.RegexpParser(grammar)
        toks = nltk.regexp_tokenize(text, sentence_re)
        postoks = nltk.tag.pos_tag(toks)
        # print (postoks)
        tree = chunker.parse(postoks)
        stopwords = Stopwords.words("english")

        def leaves(tree):
            """Finds NP (nounphrase) leaf nodes of a chunk tree."""
            for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
                yield subtree.leaves()

        def normalise(word):
            """Normalises words to lowercase and stems and lemmatizes it."""
            word = word.lower()
            # if we consider stemmer then results comes with stemmed word,
            # but in this case word will not match with comment
            # word = stemmer.stem_word(word)
            word = lemmatizer.lemmatize(word)
            return word

        def acceptable_word(word):
            """Checks conditions for acceptable word: length, stopword.
            We can increase the length if we want to consider large phrase"""
            accepted = bool(2 <= len(word) <= 40 and word.lower() not in stopwords)
            return accepted

        def get_terms(tree):
            for leaf in leaves(tree):
                term = [normalise(w) for w, t in leaf if acceptable_word(w)]
                yield term

        terms = get_terms(tree)
        self.chunks = list(terms)
        return self.chunks

    def get_all_emb(self):
        self.get_entities()
        self.get_keyphrases()
        self.get_chunks()
        self.chunks_emb = self.model.encode(
            list(itertools.chain.from_iterable(self.chunks))
        )
        self.ents_text_emb = self.model.encode(self.ents_text)
        self.keyphrases_emb = self.model.encode(self.filtered_keyphrases)

    def get_one_emb(self):
        self.chunk_emb_one = np.average(self.chunks_emb, axis=0)
        self.ents_text_emb_one = np.average(self.ents_text_emb, axis=0)
        self.keyphrases_emb_one = np.average(self.keyphrases_emb, axis=0)


# processor = ProcessText()
# processor.load_text("She left her husband. He killed their children.
# Just another day in America.")
# processor.load_text_file("test.txt")
# processor.get_entities()
# processor.get_keyphrases()
# processor.get_sentiment()
# processor.get_chunks()
# print(processor.chunks)
# processor.get_all_emb()
# processor.get_one_emb()
