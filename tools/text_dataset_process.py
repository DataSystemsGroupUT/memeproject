import pandas as pd
from tools import textprocess
from tools import meme_dataset_process
import random


class TextDatasetProcess(object):
    def __init__(
        self,
        datafile="blogtext.csv",
        nrows=50,
        lines=681285,
        meme_dataset="../datasets/Dataset Knowledge Graph.xlsx",
    ):
        t = lines - (nrows + 2)
        skip = range(1, random.randint(0, t))
        self.data = pd.read_csv(datafile, nrows=nrows, skiprows=skip)
        self.MemeProcessor = meme_dataset_process.MemeDatasetProcess(meme_dataset)
        self.MemeProcessor.processor_memes()

    def processor_articles(self):
        meme_ids = []
        for text in self.data.text:
            processed_input = textprocess.TextProcess()
            print("text: ", text)
            processed_input.load_text(text)

            # processed_input.get_sentiment()
            processed_input.get_all_emb()
            processed_input.get_one_emb()
            meme_id = self.MemeProcessor.get_similar_memes(
                processed_input, "chunk_emb_one"
            )

            meme_ids.append([text, meme_id])

        return meme_ids
