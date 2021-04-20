# you can start with TextProcess.py the main NLP code,
# the other two scripts make use of it to generate embeddings.
import pandas as pd
from scipy.spatial import distance
from tools import textprocess


class MemeDatasetProcess(object):

    def __init__(self, datafile):
        xls = datafile
        self.memes = pd.read_excel(xls, 'memes', engine='openpyxl')
        self.constructors = pd.read_excel(xls, 'constructors',
                                          engine='openpyxl')
        self.origins = pd.read_excel(xls, 'origins', engine='openpyxl')
        self.characters = pd.read_excel(xls, 'characters', engine='openpyxl')
        self.tvshows = pd.read_excel(xls, 'tvshows', engine='openpyxl')
        self.movies = pd.read_excel(xls, 'movies', engine='openpyxl')
        self.games = pd.read_excel(xls, 'games', engine='openpyxl')
        self.youtube_video = pd.read_excel(xls, 'youtube_video',
                                           engine='openpyxl')
        self.animals = pd.read_excel(xls, 'animals', engine='openpyxl')
        self.country = pd.read_excel(xls, 'country', engine='openpyxl')
        self.platform = pd.read_excel(xls, 'platform', engine='openpyxl')

    def processor_memes(self, text="title", criteria=["sentiment", "emb"]):
        self.processed_memes = []
        i = 0
        for text in self.memes[text]:
            Id = self.memes["Imgflip_ID"][i]
            processed_meme = textprocess.TextProcess()
            processed_meme.load_text(text)

            if "emb" in criteria:
                processed_meme.get_all_emb()
                processed_meme.get_one_emb()

            if "sentiment" in criteria:
                processed_meme.get_sentiment()
            self.processed_memes.append([Id, processed_meme])
            i += 1

    def get_similar_memes(self, processed_input,
                          criteria="keyphrases_emb_one"):
        if criteria == "keyphrases_emb_one":
            inp = processed_input.keyphrases_emb_one
        elif criteria == "ents_text_emb_one":
            inp = processed_input.ents_text_emb_one
        elif criteria == "chunk_emb_one":
            inp = processed_input.chunk_emb_one
        elif criteria == "all_emb_one":
            inp = processed_input.all_emb_one
        elif criteria == "sentiment":
            sent = processed_input.sentiment["compound"]

        dist = 100000000000000.0
        memeid = None

        for processed_meme in self.processed_memes:
            Id = processed_meme[0]
            processed_meme_text = processed_meme[1]
            if criteria in ["keyphrases_emb_one", "ents_text_emb_one",
                            "chunk_emb_one"]:
                if criteria == "keyphrases_emb_one":
                    meme_inp = processed_meme_text.keyphrases_emb_one
                elif criteria == "ents_text_emb_one":
                    meme_inp = processed_meme_text.ents_text_emb_one
                elif criteria == "chunk_emb_one":
                    meme_inp = processed_meme_text.chunk_emb_one

                new_dist = distance.cosine(meme_inp, inp)

                if new_dist < dist:
                    dist = new_dist
                    memeid = Id

            elif criteria == "sentiment":
                meme_sent = processed_meme_text.sentiment["compound"]
                if meme_sent < sent:
                    diff = sent - meme_sent
                else:
                    diff = meme_sent - sent

                if diff < dist:
                    dist = diff
                    memeid = Id

        return memeid
