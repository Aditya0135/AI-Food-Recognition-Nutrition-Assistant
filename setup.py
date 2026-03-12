import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

version = "0.0.1"
REPO_NAME = "AI-Food-Recognition-Nutrition-Assistant"
AUTHOR_USER_NAME = "Aditya0135"
SRC_REPO = "AI_Food_Recognition_Nutrition_Assistant"

setuptools.setup(
    name=SRC_REPO,
    version=version,
    author=AUTHOR_USER_NAME,
    description="A model which can identify the food type and predict its nutrition information",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues"
    },
    package_dir={"":"src"},
    packages=setuptools.find_packages(where="src"),
)
    