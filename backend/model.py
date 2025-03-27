# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
from pathlib import Path
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def load_recipe_data(file_path):
    """
    Load and process recipe JSON data
    """
    # Read JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Display basic information about the dataset
    print(f"\nDataset Info for {Path(file_path).name}:")
    print(f"Number of recipes: {len(df)}")
    if 'ingredients' in df.columns:
        if df['ingredients'].apply(lambda x: isinstance(x, list)).all():
            print(f"Total unique ingredients: {len(set([ing for ingredients in df['ingredients'] for ing in ingredients]))}")
        else:
            print("The 'ingredients' column does not contain lists of ingredients.")
    
    # Display sample entries
    print("\nSample Recipe:")
    print(df.iloc[0].to_dict())
    
    return df

# Assuming your JSON file is in the same folder as model.py:
json_file_path = 'test.json'

# Load the JSON data into a DataFrame
df = load_recipe_data(json_file_path) 

train_df = df
test_df = df

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
        """Build comprehensive knowledge base of ingredients"""
        for _, row in self.train_df.iterrows():
            ingredients = row['ingredients']
            cuisine = row.get('cuisine', 'unknown')
            
            # Update cuisine mappings
            self.cuisine_ingredients[cuisine].update(ingredients)
            
            # Update ingredient frequencies and cuisines
            for ingredient in ingredients:
                self.ingredient_frequencies[ingredient] += 1
                self.ingredient_cuisines[ingredient].add(cuisine)
                
                # Update ingredient contexts
                context = [ing for ing in ingredients if ing != ingredient]
                self.ingredient_contexts[ingredient].extend(context)
                
                # Update ingredient pairings
                for other_ing in context:
                    self.ingredient_pairings[ingredient][other_ing] += 1
    
    def _create_ingredient_embeddings(self):
        """Create ingredient embeddings based on their context"""
        ingredient_docs = {
            ing: ' '.join(contexts) 
            for ing, contexts in self.ingredient_contexts.items()
        }
        
        self.ingredients = list(ingredient_docs.keys())
        self.context_matrix = self.vectorizer.fit_transform(
            [ingredient_docs[ing] for ing in self.ingredients]
        )
    
    def get_ingredient_profile(self, ingredient):
        """Get detailed profile of an ingredient"""
        if ingredient not in self.ingredients:
            return None
            
        profile = {
            'frequency': self.ingredient_frequencies[ingredient],
            'cuisines': list(self.ingredient_cuisines[ingredient]),
            'top_pairings': sorted(
                self.ingredient_pairings[ingredient].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
        return profile

    def find_substitutes(self, ingredient, cuisine=None, n=10):
        """Find substitute ingredients with detailed analysis"""
        if ingredient not in self.ingredients:
            return []
            
        ing_idx = self.ingredients.index(ingredient)
        similarities = cosine_similarity(
            self.context_matrix[ing_idx:ing_idx+1], 
            self.context_matrix
        ).flatten()
        
        # Get top similar ingredients with detailed info
        most_similar = []
        sorted_indices = np.argsort(similarities)[::-1]
        
        for idx in sorted_indices:
            candidate = self.ingredients[idx]
            if candidate != ingredient:
                if cuisine and candidate not in self.cuisine_ingredients[cuisine]:
                    continue
                    
                # Calculate common pairings
                common_pairings = set(
                    self.ingredient_pairings[ingredient].keys()
                ) & set(
                    self.ingredient_pairings[candidate].keys()
                )
                
                most_similar.append({
                    'ingredient': candidate,
                    'similarity': similarities[idx],
                    'frequency': self.ingredient_frequencies[candidate],
                    'cuisines': list(self.ingredient_cuisines[candidate]),
                    'common_pairings': len(common_pairings),
                    'top_common_pairings': sorted(
                        common_pairings,
                        key=lambda x: self.ingredient_pairings[ingredient][x],
                        reverse=True
                    )[:3]
                })
                
                if len(most_similar) >= n:
                    break
        
        return most_similar

    def get_detailed_recommendations(self, ingredient, cuisine=None):
        """Get comprehensive substitute recommendations with analysis"""
        # Get ingredient profile
        profile = self.get_ingredient_profile(ingredient)
        if not profile:
            return f"Ingredient '{ingredient}' not found in database."
            
        # Get substitutes
        substitutes = self.find_substitutes(ingredient, cuisine)
        if not substitutes:
            return f"No substitutes found for {ingredient}"
            
        # Format detailed report
        report = [
            f"\nDETAILED SUBSTITUTION ANALYSIS FOR: {ingredient.upper()}",
            f"\nOriginal Ingredient Profile:",
            f"- Used in {profile['frequency']} recipes",
            f"- Found in cuisines: {', '.join(sorted(profile['cuisines']))}",
            f"- Common pairings: {', '.join(f'{ing} ({count} times)' for ing, count in profile['top_pairings'])}",
            "\nRECOMMENDED SUBSTITUTES:"
        ]
        
        for i, sub in enumerate(substitutes, 1):
            confidence = sub['similarity'] * 100
            report.extend([
                f"\n{i}. {sub['ingredient'].upper()}:",
                f"   Confidence: {confidence:.1f}%",
                f"   Frequency: Used in {sub['frequency']} recipes",
                f"   Cuisines: {', '.join(sorted(sub['cuisines']))}",
                f"   Common pairings with original: {', '.join(sub['top_common_pairings'])}"
            ])
            
        return '\n'.join(report)

    def analyze_recipe_substitutions(self, ingredients):
        """Analyze substitution possibilities for all ingredients in a recipe"""
        report = ["RECIPE SUBSTITUTION ANALYSIS:"]
        
        for ingredient in ingredients:
            substitutes = self.find_substitutes(ingredient, n=3)
            if substitutes:
                subs_text = ', '.join([
                    f"{sub['ingredient']} ({sub['similarity']*100:.1f}%)"
                    for sub in substitutes
                ])
                report.append(f"\n{ingredient}:")
                report.append(f"Top substitutes: {subs_text}")
            
        return '\n'.join(report)

def get_detailed_substitutions(train_df, ingredient):
    system = EnhancedIngredientSubstitution(train_df)
    return system.get_detailed_recommendations(ingredient)

def analyze_recipe(train_df, ingredients):
    system = EnhancedIngredientSubstitution(train_df)
    return system.analyze_recipe_substitutions(ingredients)

# Save the model to a .pkl file
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Create and save the model
system = EnhancedIngredientSubstitution(train_df)
save_model(system, 'enhanced_ingredient_substitution.pkl')

print(get_detailed_substitutions(train_df, 'water'))

recipe_ingredients = ['feta cheese', 'garlic', 'olive oil']
print(analyze_recipe(train_df, recipe_ingredients))