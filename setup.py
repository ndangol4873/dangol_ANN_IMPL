from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

    USERNAME = "ndangol4873"
    PROJECT_NAME = "dangol_ANN_IMPL"

setup(
    name="src",
    version="0.0.1",
    author=USERNAME,
    description="A small package for ANN Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/{USERNAME}/{PROJECT_NAME}",
    author_email="nareshkadambini@gmail.com",
    packages=["src"],
    python_requires=">=3.7",
    install_requires=[
        "tensorflow",
        "matplotlib",
        "seaborn",
        "numpy",
        "pandas"
    ]
)
