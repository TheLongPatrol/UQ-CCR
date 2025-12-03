# Define the mapping of article numbers to news sources in a list
# Index 0 corresponds to article1.txt, index 1 to article2.txt, and so on
import os
sources_list = [
    "Reuters",
    "Vox",
    "Financial Times",
    "Barron's",
    "Barron's_2",
    "Investor's Business Daily",
    "WSJ",
    "The Sun",
    "The Sun_2",
    "AP",
    "Business Insider",
    "Cryptonews",
    "Investopedia",
    "The Times",
    "MarketWatch",
    "PCMag",
    "The Daily Hodl",
    "Fortune",
    "Forbes",
    "BBC"
]

# Function to map article filenames to their corresponding news source
def map_article_to_source(article_name):
    # Extract the article number from the filename (e.g., article1.txt -> 1)
    article_number = int(article_name.replace("article", "").replace(".txt", ""))
    
    # The list is 0-indexed, so we subtract 1 from the article number
    if 1 <= article_number <= len(sources_list):
        return sources_list[article_number - 1]
    else:
        return "Unknown Source"

# Function to convert article filenames to their corresponding news sources
def convert_to_news_source(article_names, file_dir):
    for file_name in article_names:
        article_name, ext = file_name.split(".")
        article_number = int(article_name.replace("article", "").replace("doc", "").replace(ext, ""))

        source = sources_list[article_number - 1]
        os.rename(file_dir+file_name, file_dir+source+".txt")

# Example Usage
file_dir = "bitcoin_docs/"
article_files = os.listdir(file_dir)
news_sources = convert_to_news_source(article_files, file_dir)

file_dir = "relations/"
article_files = os.listdir(file_dir)
news_sources = convert_to_news_source(article_files, file_dir)

file_dir = "relations_json/"
article_files = os.listdir(file_dir)
news_sources = convert_to_news_source(article_files, file_dir)
