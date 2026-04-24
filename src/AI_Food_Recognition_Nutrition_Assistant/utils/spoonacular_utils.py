import requests
import streamlit as st
from typing import Optional, Dict, List

try:
    from api_config import SPOONACULAR_API_KEY
except ImportError:
    SPOONACULAR_API_KEY = None

API_BASE_URL = "https://api.spoonacular.com"


def get_api_key():
    """Get API key from config or Streamlit secrets."""
    if SPOONACULAR_API_KEY and SPOONACULAR_API_KEY != "YOUR_API_KEY_HERE":
        return SPOONACULAR_API_KEY

    # Try to get from Streamlit secrets
    try:
        return st.secrets.get("SPOONACULAR_API_KEY")
    except:
        return None


def search_recipe_and_nutrition(food_name: str):
    """
    Search for food recipe with nutrition.
    """
    try:
        url = f"{API_BASE_URL}/recipes/complexSearch"
        params = {
            "query": food_name,
            "number": 1,
            "ranking": "popularity",
            "apiKey": "a03c1ff555034e0aaf14a34ba5d93d92",
            "addRecipeNutrition": True,
        }

        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()

        data = response.json()
        return data["results"][0]

    except Exception as e:
        print(f"❌ Recipe search error: {str(e)}")
        return None


def extract_macro_nutrients(recipe_data: Dict) -> List[Dict]:
    """
    Extract macro nutrients from recipe data.
    
    Args:
        recipe_data: Recipe data containing nutrition information
        
    Returns:
        List of dictionaries with nutrient names and values
    """
    try:
        nutrients = recipe_data.get('nutrition', {}).get('nutrients', [])
        nutrients_to_extract = ['Calories', 'Carbohydrates', 'Protein', 'Fat', 'Sugar', 'Fiber']
        macro_nutrients = [
            {n['name']: f"{n['amount']}{n['unit']}"} 
            for n in nutrients 
            if n['name'] in nutrients_to_extract
        ]
        return macro_nutrients
    except Exception as e:
        print(f"❌ Error extracting macro nutrients: {str(e)}")
        return []


def get_full_recipe_info(recipe_id: int) -> Optional[Dict]:
    """
    get recipe info links and procedure
    """
    try:
        ID = recipe_id
        url = f"{API_BASE_URL}/recipes/{ID}/information"
        params = {
            "apiKey": "a03c1ff555034e0aaf14a34ba5d93d92",
        }
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()

        data = response.json()
        return data
    except Exception as e:
        print(f"❌ Error fetching recipe info: {str(e)}")
        return None


def extract_recipe(data: Dict) -> Dict:
    """
    Extract recipe details from full recipe info.
    
    Args:
        data: Full recipe information from API
        
    Returns:
        Dictionary with recipe source, title, time, ingredients, and instructions
    """
    try:
        return {
            "source_url": data.get("sourceUrl"),
            "title": data.get("title"),
            "ready_in_minutes": data.get("readyInMinutes"),
            "servings": data.get("servings", 1),
            "ingredients": [i["original"] for i in data.get("extendedIngredients", [])],
            "instructions": [
                s["step"] for s in data.get("analyzedInstructions", [{}])[0].get("steps", [])
            ]
        }
    except Exception as e:
        print(f"❌ Error extracting recipe: {str(e)}")
        return {}


def format_nutrition_data(nutrition_list: List) -> Dict:
    """
    Format nutrition list into a dictionary with macro nutrients.
    
    Args:
        nutrition_list: List of nutrition items from API
        
    Returns:
        Dictionary with formatted nutrition data
    """
    try:
        nutrition_dict = {}
        for nutrient in nutrition_list:
            name = nutrient.get('name', '').lower()
            amount = nutrient.get('amount', 0)
            unit = nutrient.get('unit', '')
            
            # Map nutrient names to common keys
            if 'calorie' in name:
                nutrition_dict['calories'] = {'amount': amount, 'unit': unit}
            elif 'protein' in name:
                nutrition_dict['protein'] = {'amount': amount, 'unit': unit}
            elif 'carb' in name:
                nutrition_dict['carbs'] = {'amount': amount, 'unit': unit}
            elif 'fat' in name and 'saturated' not in name:
                nutrition_dict['fat'] = {'amount': amount, 'unit': unit}
            elif 'fiber' in name:
                nutrition_dict['fiber'] = {'amount': amount, 'unit': unit}
            elif 'sugar' in name:
                nutrition_dict['sugar'] = {'amount': amount, 'unit': unit}
        
        return nutrition_dict
    except Exception as e:
        print(f"❌ Error formatting nutrition data: {str(e)}")
        return {}


def get_food_data(food_name: str) -> Dict:
    """
    Get complete food data including nutrition and recipe information.
    
    This is the main function that orchestrates all API calls to fetch
    comprehensive food information.
    
    Args:
        food_name: Name of the food to search for
        
    Returns:
        Dictionary with structure:
        {
            "nutrition": {...},
            "recipe": {...}
        }
    """
    try:
        api_key = get_api_key()
        if not api_key:
            print("❌ No API key found")
            return {"nutrition": {}, "recipe": {}}
        
        # Step 1: Search for recipe with nutrition
        recipe_search = search_recipe_and_nutrition(food_name)
        if not recipe_search:
            print(f"❌ No recipe found for {food_name}")
            return {"nutrition": {}, "recipe": {}}
        
        recipe_id = recipe_search.get('id')
        nutrition_data = recipe_search.get('nutrition', {})
        
        # Step 2: Get full recipe information
        full_recipe_info = get_full_recipe_info(recipe_id)
        if not full_recipe_info:
            print(f"❌ Could not fetch full recipe info for ID {recipe_id}")
            full_recipe_info = {}
        
        # Step 3: Extract and format recipe details
        recipe_details = extract_recipe(full_recipe_info)
        
        # Step 4: Format nutrition data
        nutrients_list = nutrition_data.get('nutrients', [])
        nutrition_formatted = format_nutrition_data(nutrients_list)
        
        # Combine and return
        result = {
            "nutrition": nutrition_formatted,
            "recipe": recipe_details
        }
        
        return result
        
    except Exception as e:
        print(f"❌ Error getting food data: {str(e)}")
        return {"nutrition": {}, "recipe": {}}

