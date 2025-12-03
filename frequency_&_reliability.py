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
    "Cosmopolitan": 24.22,
    "The Washington Free Beacon": 24.05,
    "Bill O'Reilly": 24.16,
    "Daily Dot": 25.56,
    "Townhall": 25.13,
    "The Root": 26.51,
    "Daily Kos": 26.75,
    "Media Matters for America": 26.45,
    "Rolling Stone": 26.24,
    "Mirror - UK": 26.88,
    "Independent Journal Review | IJR": 26.13,
    "RealClearPolitics": 26.90,
    "Published Reporter": 26.86,
    "Right Wing Watch | People For the American Way": 27.23,
    "Rebel News": 27.12,
    "OutKick": 27.30,
    "Jezebel": 28.55,
    "RT Russia Today": 28.71,
    "The Spectator World": 28.42,
    "Newsmax (website)": 28.14,
    "OAN Network (website)": 28.56,
    "Heritage Foundation": 28.68,
    "The Nation": 30.81,
    "New Republic": 30.03,
    "Daily Mail": 30.80,
    "Breitbart": 30.40,
    "Blaze Media": 30.18,
    "Democracy Now! (website)": 31.68,
    "Daily Beast": 31.22,
    "New York Post": 31.35,
    "The Post Millennial": 31.43,
    "New York Sun": 31.50,
    "National Review": 31.32,
    "Daily Wire": 31.11,
    "The American Conservative": 31.64,
    "Salon": 32.93,
    "Hill Reporter": 32.56,
    "Blavity News": 32.61,
    "Inquisitr": 32.92,
    "Just the News": 32.56,
    "National Catholic Register": 32.13,
    "In These Times": 33.27,
    "Truthout": 33.35,
    "Vanity Fair": 33.34,
    "Raw Story": 33.99,
    "Washington Times": 33.46,
    "CBN": 34.00,
    "MS NOW | MSNBC (website)": 34.52,
    "Mediaite": 34.41,
    "The Bulwark": 34.17,
    "TMZ": 34.73,
    "The Epoch Times": 34.45,
    "The College Fix": 34.39,
    "Fox News (website)": 34.91,
    "Washington Examiner": 34.06,
    "Daily Caller": 34.43,
    "Rasmussen Reports": 34.05,
    "Mother Jones": 35.61,
    "Slate": 35.29,
    "AlterNet": 35.81,
    "Washington Monthly": 35.78,
    "Talking Points Memo - John Marshall": 35.81,
    "Reason": 35.79,
    "City Journal": 35.00,
    "The Lever - David Sirota": 36.18,
    "Newser": 36.83,
    "Fortune": 36.91,
    "Boston Herald": 36.31,
    "The Christian Post": 36.16,
    "Common Dreams": 37.41,
    "Teen Vogue": 37.79,
    "The Intercept": 37.39,
    "WIRED": 37.13,
    "Quillette": 37.53,
    "American Independent": 38.38,
    "HuffPost": 38.06,
    "The Week": 38.39,
    "The Atlantic": 38.04,
    "The Verge": 38.10,
    "The Independent": 38.49,
    "Washington Post": 38.67,
    "Fox Business (website)": 38.66,
    "Advocate Magazine": 39.45,
    "Vogue": 39.13,
    "TheGrio": 39.55,
    "Vox": 39.65,
    "Miami Herald": 39.02,
    "Upworthy": 39.86,
    "Newsweek": 39.14,
    "Straight Arrow News": 39.89,
    "The New Yorker": 40.85,
    "The Guardian": 40.18,
    "Forward": 40.45,
    "The New York Times": 40.97,
    "Business Insider": 40.61,
    "MarketWatch": 40.93,
    "USA Today": 40.91,
    "Forbes": 40.70,
    "CNET": 40.02,
    "American Action Forum": 40.28,
    "Deseret News": 40.70,
    "The Center Square": 40.32,
    "Orange County Register": 40.58,
    "Time Magazine": 41.07,
    "Al Jazeera (website)": 41.29,
    "Boston Globe": 41.94,
    "Poynter": 41.75,
    "People Magazine": 41.58,
    "CBS News (website)": 41.90,
    "The Economist": 41.67,
    "The Hill": 41.65,
    "Sky News": 41.77,
    "Christianity Today": 41.75,
    "The Dispatch": 41.08,
    "World News Group": 41.51,
    "CNN (website)": 42.10,
    "LA Times": 42.76,
    "NBC News (website)": 42.74,
    "Politico": 42.31,
    "Foreign Policy": 42.63,
    "Bloomberg News": 42.24,
    "NewsNation (website)": 42.94,
    "Foreign Affairs": 43.54,
    "Columbia Journalism Review": 43.33,
    "Detroit Free Press": 43.46,
    "NPR (website)": 43.09,
    "PBS": 43.30,
    "Fiscal Times": 43.33,
    "Axios": 43.10,
    "US News and World Report": 43.55,
    "Scripps News (website)": 43.59,
    "Dallas Morning News": 43.65,
    "CNBC (website)": 43.75,
    "The Globe and Mail": 43.99,
    "Tangle": 43.04,
    "Wall Street Journal": 43.30,
    "Financial Times": 44.00,
    "Politifact": 44.67,
    "ABC News (website)": 44.54,
    "AP | Associated Press": 44.80,
    "Christian Science Monitor": 44.23,
    "BBC": 44.66,
    "Barron's": 44.78,
    "Reuters": 45.00,
    "The Marshall Project": 45.76,
    "Chicago Sun-Times": 45.44,
    "11Alive Atlanta NBC WXIA": 45.75,
    "UPI": 45.32,
    "Arkansas Democrat-Gazette": 45.14,
    "Patch": 46.09,
    "The Weather Channel": 46.20,
    "Newsday": 46.71,
    "BNO News": 46.37,
    "Chron.com": 46.63,
    "Air Force Times": 46.01,
    "ProPublica": 47.22,
    "Agence France-Presse | AFP": 47.15,
    "USAFacts": 50.32
}

for source in retrieved["sources appeared in"]:
    print(reliability_map[source])