import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="drbert",
    version="0.0.1",
    author="John Giorgi",
    author_email="johnmgiorgi@gmail.com",
    description="Dr.BERT: Multi-task approach to clinical NLP based on pre-trained language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JohnGiorgi/drbert",
    packages=setuptools.find_packages(),
    keywords=['clinical natural language processing', 'transformers', 'BERT', 'Dr. BERT'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch>=1.2.0',
        'torchtext>=0.4.0',
        'tensorboard>=1.14.0',
        'transformers>=2.0.0',
        'spacy>=2.1.4',
        'Keras-Preprocessing>=1.1.0',
        'nltk>=3.3',
        'PTable>=0.9.2',
        'xmltodict>=0.12.0',
        'seqeval>=0.0.12',
        'scikit-learn>=0.21.2'
    ],
    extras_require={
        'dev': [
            'tox',
            'pytest',
            'pytest-cov'
        ]
    }
)
