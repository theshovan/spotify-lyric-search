import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SpotifyLyricSearch:
    def __init__(self, data_path):
        # Load the provided dataset
        print(f"Loading dataset from: {data_path}...")
        self.df = pd.read_csv(data_path)
        
        # Initialize Vectorizer with built-in English stop-word removal 
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        self.tfidf_matrix = None
        self._initialize_engine()

    def _preprocess(self, text):
        """Standardizes text: lowercase and removes non-alphabetic characters """
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text

    def _initialize_engine(self):
        """Preprocesses the entire dataset and builds the TF-IDF matrix """
        print("Preprocessing lyrics and building index...")
        processed_lyrics = self.df['text'].apply(self._preprocess)
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_lyrics)
        print("Engine Ready.")

    def identify(self, snippet):
        """Identifies the Song Title and Artist given a text snippet """
        cleaned_snippet = self._preprocess(snippet)
        snippet_vec = self.vectorizer.transform([cleaned_snippet])
        
        # Calculate Cosine Similarity 
        # Formula: $cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$
        similarities = cosine_similarity(snippet_vec, self.tfidf_matrix).flatten()
        
        # Retrieve the best match
        best_idx = similarities.argmax()
        
        return {
            "song": self.df.iloc[best_idx]['song'],
            "artist": self.df.iloc[best_idx]['artist'],
            "similarity_score": round(similarities[best_idx], 4)
        }

    def run_accuracy_test(self, n=20):
        """Demonstrates model accuracy by testing random samples """
        samples = self.df.sample(n)
        correct = 0
        
        for _, row in samples.iterrows():
            # Create a 10-word snippet from the actual lyrics
            lyric_words = row['text'].split()
            snippet = " ".join(lyric_words[20:30]) if len(lyric_words) > 30 else " ".join(lyric_words[:10])
            
            result = self.identify(snippet)
            if result['song'] == row['song']:
                correct += 1
        
        accuracy = (correct / n) * 100
        print(f"\nDemonstration Accuracy: {accuracy}% on {n} samples.")

# Execution
if __name__ == "__main__":
    # Use the actual fileName of the provided dataset
    FILE_NAME = 'Spotify Million Song Dataset_exported.csv (1).zip/Spotify Million Song Dataset_exported.csv'
    
    engine = SpotifyLyricSearch(FILE_NAME)
    
    # Example Identification
    test_snippet = "Look at her face it's a wonderful face and it means something special"
    prediction = engine.identify(test_snippet)
    
    print("\n--- Model Prediction ---")
    print(f"Input Snippet: \"{test_snippet}\"")
    print(f"Identified Song: {prediction['song']}")
    print(f"Artist: {prediction['artist']}")
    
    # Demonstration of Accuracy 
    engine.run_accuracy_test(n=10)