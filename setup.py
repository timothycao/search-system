from setuptools import setup, find_packages

setup(
    name="search_system",
    version="1.0.0",
    packages=find_packages(include=["search_system", "search_system.*"]),
    python_requires=">=3.8",
)