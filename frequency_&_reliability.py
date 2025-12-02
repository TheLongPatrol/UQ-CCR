from collections import defaultdict

class SentenceIndexer:
    def __init__(self):
        self.index = defaultdict(list)

    def load_from_file(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = line.split(',')
                    if len(parts) < 2: continue
                    source_name = parts[0].strip()
                    raw_sentences = parts[1:]
                    clean_sentences = [s.strip() for s in raw_sentences if s.strip()]
                    self._add_to_index(source_name, clean_sentences)

            print(f"Successfully loaded and indexed data from {filepath}")

        except FileNotFoundError:
            print(f"Error: Could not find file {filepath}")

    def _add_to_index(self, source, sentences):
        unique_sentences = set(sentences)
        for sentence in unique_sentences:
            self.index[sentence].append(source)

    def search(self, query_sentence):
        query_sentence = query_sentence.strip()

        if query_sentence in self.index:
            sources = self.index[query_sentence]
            return {
                "sentence": query_sentence,
                "frequency": len(sources),
                "sources appeared in": sources
            }
        else:
            return f"Sentence not found: '{query_sentence}'"

if __name__ == "__main__":
    indexer = SentenceIndexer()
    indexer.load_from_file("temp.txt") # Reads a text file where each line is: Source website, concatenated triple sentence 1, concatenated triple sentence 2...

    query = "Relatively robust U.S. data supported sentiment in equity markets"
    retrieved = indexer.search(query)
    print(retrieved)

reliability_map = {
    "source1": 9.5,
    "source2": 9.8,
    "source3": 8.0
}

for source in retrieved["sources appeared in"]:
    print(reliability_map[source])