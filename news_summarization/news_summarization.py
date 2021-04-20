# import nltk
from textblob import TextBlob
from newspaper import Article


class NewsSummarization(object):
    """this class will summarize an article which provided by an URL

    Args:
        object ([type]): [an article ]
    """

    def __init__(self, url):
        self.url = url

    def summarizer(self):
        """summarize an article by using Newspaper package.

        Returns:
            [title]: [title of article]
            [summary]: [summary of article]
            [publish_date]: [publish date of article]
            [keywords]: [keywords of article]
            [polarity]: [polarity is used for finding sentiment of article]
        """
        try:
            article = Article(str(self.url))
            article.download()
            article .parse()
            article .nlp()

            title = article .title
            summary = article .summary
            publish_date = article .publish_date
            keywords = article.keywords

            analysis = TextBlob(summary)
            polarity = analysis.polarity

            return title, summary, publish_date, keywords, polarity
        except Exception:
            print('An exception occurred')
