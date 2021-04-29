from news_summarization import NewsSummarization


if __name__ == '__main__':
    # Please provide the URL of an article as input.
    url = input("Please provide the URL of an article as input:")
    url = """https://news.err.ee/1608184081/restrictions-to-be-relaxed-from-april-26"""
    news_summarization = NewsSummarization(url)
    title, summary, publish_date, keywords, polarity = news_summarization.summarizer()

    print("######################################################################")
    print(f"Title: {title}")
    print("######################################################################")
    print(f"Publish date: {publish_date}")
    print("######################################################################")
    print(f"Summary: {summary}")
    print("######################################################################")
    print(f"keywords: {keywords}")
    print("######################################################################")
    print(f"polarity: {polarity}")
    print("######################################################################")
    print(f'Sentiment:{ "positive" if polarity > 0 else "negative" if polarity < 0 else "neutral" }')
    print("######################################################################")