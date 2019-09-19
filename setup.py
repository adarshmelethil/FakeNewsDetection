#!/usr/bin/env python
from setuptools import setup, find_packages
from os import environ

setup(
  name="FakeNewsDetection",
  version="0.0.1",
  package_dir={"": "src"},
  packages=find_packages("src"),
  include_package_data=True,
  install_requires=[
    "spacy"
  ],
  entry_points = {
    'console_scripts': [
      'DataCup=FakeNewsDetection.pipeline:main',
      'DataCupLoad=FakeNewsDetection.raw_data.data:main',
      'DataCup_test_search=FakeNewsDetection.search:main',
      'DataCup_test_roberta=FakeNewsDetection.roberta:main'
    ],
  }
)
