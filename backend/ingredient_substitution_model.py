import numpy as np
import pandas as pd
import json
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def load_recipe_data(file_path):
    """Load and process recipe JSON data."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def defaultdict_int():
    return defaultdict(int)

class EnhancedIngredientSubstitution:
    def __init__(self, train_df):
        self.train_df = train_df
        self.ingredient_contexts = defaultdict(list)
        self.cuisine_ingredients = defaultdict(set)
        self.ingredient_frequencies = defaultdict(int)
        self.ingredient_cuisines = defaultdict(set)
        self.ingredient_pairings = defaultdict(defaultdict_int)
        self.vectorizer = TfidfVectorizer()
        self._build_knowledge_base()
        self._create_ingredient_embeddings()
    
    def _build_knowledge_base(self):
        for _, row in self.train_df.iterrows():
            ingredients = row['ingredients']
            cuisine = row.get('cuisine', 'unknown')
            self.cuisine_ingredients[cuisine].update(ingredients)
            for ingredient in ingredients:
                self.ingredient_frequencies[ingredient] += 1
                self.ingredient_cuisines[ingredient].add(cuisine)
                context = [ing for ing in ingredients if ing != ingredient]
                self.ingredient_contexts[ingredient].extend(context)
                for other_ing in context:
                    self.ingredient_pairings[ingredient][other_ing] += 1
    
    def _create_ingredient_embeddings(self):
        ingredient_docs = {ing: ' '.join(contexts) for ing, contexts in self.ingredient_contexts.items()}
        self.ingredients = list(ingredient_docs.keys())
        self.context_matrix = self.vectorizer.fit_transform([ingredient_docs[ing] for ing in self.ingredients])
    
    def find_substitutes(self, ingredient, cuisine=None, n=10):
        if ingredient not in self.ingredients:
            return []
        ing_idx = self.ingredients.index(ingredient)
        similarities = cosine_similarity(self.context_matrix[ing_idx:ing_idx+1], self.context_matrix).flatten()
        most_similar = []
        sorted_indices = np.argsort(similarities)[::-1]
        for idx in sorted_indices:
            candidate = self.ingredients[idx]
            if candidate != ingredient:
                if cuisine and candidate not in self.cuisine_ingredients[cuisine]:
                    continue
                most_similar.append(candidate)
                if len(most_similar) >= n:
                    break
        return most_similar

def save_model(model, filename):
    """Save the trained model as a pickle file."""
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    """Load the trained model from a pickle file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)

if __name__ == "__main__":
    json_file_path = 'test.json'
    df = load_recipe_data(json_file_path)
    system = EnhancedIngredientSubstitution(df)
    save_model(system, 'enhanced_ingredient_substitution.pkl')
