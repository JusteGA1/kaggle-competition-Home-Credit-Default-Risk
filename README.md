# Kaggle competition "Home Credit Default Risk"

Full notebook and journey on kaggle competition https://www.kaggle.com/c/home-credit-default-risk/overview   
Notebook includes some basic EDA, baseline modelling, feature engineering, merge of files and final modelling with various experiments such as hyperparameters optimization, data sampling, etc.

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Status](#status)
* [License](#license)
* [Inspiration](#inspiration)
* [Contact](#contact)

## General info
Participation at kaggle competition was for learning purposes.  

Competition is for solving binary classification task on loan repayment (1 - client with payment difficulties, 0 - all other cases). There are 7 different files and the main file contains information about applied person's gender, some possessions, family, occupancy, residence and application information such as weekday, time, provided documents, etc. Other files contain information from bureau, credit card, previous applications, and more.

My personal goal was to try out several different models and understand their principles, parameters and speed. As well, I wanted to work with a bigger scope of data and several different files in order to polish my skills in concating, merging and analyzing data.  In the notebook you will find a lot of experiments and their results with relevant conclusions, and although I cared about data, my main focus was on modelling and scoring at kaggle. 

In order to shrink the notebook, some functions which only provides technical information are displaced on external python files.

## Technologies
* Python - version 3.8
* Jupyter Notebook 
* scikit-learn tools
* LightGBM
* XGBClassifier
* CatBoostClassifier
   

## Status
Project is: _finished_

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Inspiration
Learning @Turing College

## Contact
Created by [Juste Gaviene](mailto:juste.gaviene@gmail.com?subject=[GitHub]%20Source%20Han%20Sans)
