from tools import TextProcess


if __name__ == "__main__":
    processor = TextProcess.TextProcess()
    # processor = ProcessText()
    processor.load_text("She left her husband. He killed their children. Just another day in America.")
    # processor.load_text_file("test.txt")
    processor.get_entities()
    processor.get_keyphrases()
    processor.get_sentiment()
    processor.get_chunks()
    print(processor.chunks)
    # #processor.get_all_emb()
    # #processor.get_one_emb()
    print(processor.get_sentiment())
    print("Done!")
