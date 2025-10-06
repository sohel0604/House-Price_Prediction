from setuptools import find_packages, setup

# Define the project details
setup(
    name='HousePricePrediction',
    version='0.0.1',
    author='Sohel Kumar Sahoo',
    author_email='work.sksahoo@gmail.comm', 
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'catboost',
        'xgboost',
        'dill',
        'streamlit'
    ]
)
