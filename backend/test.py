import pickle
from model import EnhancedIngredientSubstitution  # Import the class definition

def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Load the model from the .pkl file
loaded_model = load_model('enhanced_ingredient_substitution.pkl')

# Test the model with a new ingredient
new_ingredient = 'tomato'
detailed_substitutions = loaded_model.get_detailed_recommendations(new_ingredient)
print(detailed_substitutions)

# Test the model with a new recipe
new_recipe_ingredients = ['chicken', 'lemon', 'thyme']
recipe_analysis = loaded_model.analyze_recipe_substitutions(new_recipe_ingredients)
print(recipe_analysis)