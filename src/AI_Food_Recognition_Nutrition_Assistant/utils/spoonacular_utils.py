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


@st.cache_data(ttl=3600)
def search_food_recipe(food_name: str, api_key: str) -> Optional[Dict]:
    """
    Search for food recipe with EVERYTHING in ONE API call:
    recipe, nutrition, ingredients, instructions.

    Args:
        food_name: Name of the food (e.g., "pizza")
        api_key: Spoonacular API key

    Returns:
        dict: Complete recipe data with nutrition, ingredients, and instructions
    """
    try:
        url = f"{API_BASE_URL}/recipes/complexSearch"
        params = {
            "query": food_name,
            "number": 1,
            "ranking": "popularity",
            "apiKey": api_key,
            "addRecipeInformation": True,
            "addRecipeInstructions": True,
            "addRecipeNutrition": True,
            "includeIngredients": True,
        }

        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()

        data = response.json()
        if data.get("results"):
            return data["results"][0]
        return None

    except Exception as e:
        print(f"❌ Recipe search error: {str(e)}")
        return None


def format_recipe(recipe_data: Dict) -> Dict:
    """
    Format recipe data for display.

    Args:
        recipe_data: Raw recipe data from API

    Returns:
        dict: Formatted recipe with ingredients and steps
    """
    if not recipe_data:
        return {"error": "No recipe data found."}

    try:
        ingredients = []
        if recipe_data.get("extendedIngredients"):
            for ing in recipe_data["extendedIngredients"]:
                ingredients.append({
                    "name": ing.get("name", ""),
                    "amount": ing.get("measures", {}).get("metric", {}).get("amount", 0),
                    "unit": ing.get("measures", {}).get("metric", {}).get("unitShort", "")
                })

        # Get cooking steps
        steps = []
        if recipe_data.get("analyzedInstructions"):
            for instruction_group in recipe_data["analyzedInstructions"]:
                if instruction_group.get("steps"):
                    for step in instruction_group["steps"]:
                        steps.append(step.get("step", ""))

        return {
            "title": recipe_data.get("title", ""),
            "image": recipe_data.get("image", ""),
            "servings": recipe_data.get("servings", 1),
            "ready_in_minutes": recipe_data.get("readyInMinutes", 0),
            "ingredients": ingredients,
            "instructions": steps,
            "source_url": recipe_data.get("sourceUrl", ""),
            "recipe_id": recipe_data.get("id", 0)
        }

    except Exception as e:
        print(f"❌ Recipe formatting error: {str(e)}")
        return {"error": str(e)}


def extract_nutrition(recipe_data: Dict) -> Optional[Dict]:
    """
    Extract nutrition data from recipe response.

    Args:
        recipe_data: Recipe data with nutrition field

    Returns:
        dict: Nutrition data (calories, protein, carbs, fat, fiber)
    """
    try:
        if not recipe_data or not recipe_data.get("nutrition"):
            return None

        nutrition = recipe_data["nutrition"]
        return {
            "calories": round(nutrition.get("calories", 0), 1),
            "protein": round(nutrition.get("protein", 0), 1),
            "carbs": round(nutrition.get("carbohydrates", 0), 1),
            "fat": round(nutrition.get("fat", 0), 1),
            "fiber": round(nutrition.get("fiber", 0), 1),
        }

    except Exception as e:
        print(f"❌ Nutrition extraction error: {str(e)}")
        return None


def get_food_data(food_name: str) -> Dict:
    """
    Get complete food data: recipe + nutrition (ONE API call).

    Args:
        food_name: Name of the food

    Returns:
        dict: Combined recipe and nutrition data
    """
    api_key = get_api_key()

    if not api_key:
        return {
            "error": "⚠️ Spoonacular API key not configured. Add it to api_config.py",
            "recipe": None,
            "nutrition": None
        }

    # ONE API CALL gets everything
    recipe_raw = search_food_recipe(food_name, api_key)

    if recipe_raw:
        recipe = format_recipe(recipe_raw)
        nutrition = extract_nutrition(recipe_raw)
        return {
            "recipe": recipe,
            "nutrition": nutrition,
            "error": None
        }
    else:
        return {
            "recipe": None,
            "nutrition": None,
            "error": f"No recipe found for '{food_name}'"
        }

