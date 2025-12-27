from setuptools import setup, find_packages

setup(
    name="blockchain_fraud_prevention",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "numpy>=1.24.0"
    ],
    author="Santiago Torterolo",
    author_email="santiago@example.com",
    description="Blockchain fraud detection system with ML",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/santiago-torterolo/blockchain_fraud_prevention",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
