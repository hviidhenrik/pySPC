from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyspc",
    version="1.0",
    author="Henrik Hviid Hansen, Sebastian Olivier Nymann Topalian, Davide Cacciarelli",
    author_email="hehha@orsted.com, sebtop@kt.dtu.dk, dcac@dtu.dk",
    description="Plug and play statistical process control functions for Python",
    long_description=long_description,
    long_description_content_type="markdown",
    url="https://github.com/hviidhenrik/pySPC",
    packages=find_packages(exclude=["tests*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "scikit_learn",
        "scipy",
        "statsmodels",
        "torch",
        "openpyxl",
        "seaborn"
    ],
)
