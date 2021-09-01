from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open("requirements.txt", mode="r", encoding="utf-8") as requirements_file:
    requirements = requirements_file.read()

setup(
    name='morpheus_multilingual',
    version="1.0.0",
    packages=find_packages(),
    description="morpheus_multilingual: "
                "A Study of Morphological Robustness of Neural Machine Translation",
    long_description=readme,
    install_requires=requirements,
    url="https://github.com/murali1996/morpheus_multilingual",
)
