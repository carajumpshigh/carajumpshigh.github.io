# Portfolio
---
## Data Science

[**Pipeline Leakage Diagnoses: how can we response in time and reduce the impact of lifeline pipe network accidents?**](https://github.com/carajumpshigh/Pipeline_Leakage_Diagnosis_based_on_Data_Monitoring_and_Machine_Learning)

[![Initial Anomaly Detection Model](https://img.shields.io/badge/Python-%20Initial%20Anomaly%20Detection%20Model-blue?logo=Python)](https://github.com/carajumpshigh/Pipeline_Leakage_Diagnosis_based_on_Data_Monitoring_and_Machine_Learning/blob/master/main_1.py)
[![Optimized Model with Geospatial Data](https://img.shields.io/badge/Python-%20Optimized%20Model%20with%20Geospatial%20Data-blue?logo=Python)](https://github.com/carajumpshigh/Pipeline_Leakage_Diagnosis_based_on_Data_Monitoring_and_Machine_Learning/blob/master/main_2.py)

<div> The purpose of this data mining project is to builds a leakage detection model based on data analysis of the multi-source data of the existing gas pipeline network monitoring system(SCADA) of Suzhou, China, and then establishes the risk early warning system of the leakage of each pipeline segment, so as to response in time and reduce the impact of lifeline pipe network accidents.
</div>
<br>
For the initial anomaly detection model, I plot the line charts to show the daily trend of the pipeline pressure of the SCADA measuring points, and use Dynamic time warping(DTW) algorithm to measure the similarity of them. Then I do clustering to generate typical modes of the daily trends and find the risky modes with most pipeline lackage happening. Specifically, I generate double-layer clustering to develop the efficiency of this model by reducing 98% of the total calculation, considering the huge dataset(over 16,000 records).
<br>
<center><img src="images/Leakage Diagnosis Model.png"/></center>
With geospatial data of the historical leakage accidents, I further improved the efficiency of the model by defining the risk clusters, with the idea that the pipelines are affected by their "neighbors", and there are more important pipelines(e.g. with more connecting pipes) that can be used to represent the mode of their "adjunctive" pipes in the risk cluster. I identify the representative pipelines with Page Rank algorithm and further reduce the runtime of our model by adding another layer of clustering.
<center><img src="images/Leakage Diagnosis Model with Spatial Info.png"/></center>
<br>
Models: Anomaly detection, DTW, Spectral clustering, Page Rank
<br>

---
[**Analysis of NYC 311 Noise Complaints: Fun Facts about Noise in NYC**](https://github.com/carajumpshigh/Analysis_of_NYC_Noise_Complaints)

[![Time series analysis](https://img.shields.io/badge/Jupyter-%20Time%20Series%20Analysis-blue?logo=Jupyter)](https://github.com/carajumpshigh/Analysis_of_NYC_Noise_Complaints/blob/master/Noise_Complaints_Time_Analysis.ipynb)
[![Spatial analysis](https://img.shields.io/badge/Jupyter-%20Spatials%20Analysis-blue?logo=Jupyter)](https://github.com/carajumpshigh/Analysis_of_NYC_Noise_Complaints/blob/master/Noise%20Category%20Analysis.ipynb)
[![Results](https://img.shields.io/badge/Images-%20Results-blue?logo=Github)](https://github.com/carajumpshigh/Analysis_of_NYC_Noise_Complaints/tree/master/result_img)

<div> This project analyzes NYC 311 noise complaints and sees how the complaints of differet kinds of noise distribute geographically and change by the time of day/year. After feature engineering with PCA, regression is applied to study the correlation between noise complaints and demographic, economic and socio attributes in different neighborhoods.
</div>
<br>
Some interesting findings: 1) Noise complaints are reported more frequently in summer; 2) While in most part of New York, the main reasons of noise complaints is loud music and parties, people in Manhattan, Brooklyn Height and Red Hook tend to report more about construction noise, and people in Upper Bronx are mostly annoyed by ice cream trucks; 3) Education and Origin factors take the lead in related factors, followed by Salary and Race.<br>
</div>
<center><img src="images/Workflow_NYCNoise_model.jpg"/></center> 
<br>
Models: Time series analysis, Spatial joint, PCA, Regression
<br>
  
---
[**Product Differentiation in the Automobiles Market: An Empirical Analysis**](https://github.com/Emmyphung/car_models/blob/master/README.md)

[![EDA](https://img.shields.io/badge/Jupyter-Stock_analysis_with_interative_charts-blue?logo=Jupyter)](https://github.com/Emmyphung/car_models/blob/master/car_EDA.ipynb)
[![Models](https://img.shields.io/badge/Jupyter-Stock_prediction-blue?logo=Jupyter)](https://github.com/Emmyphung/car_models/blob/master/car_modelling.ipynb)

<div> This research project examined the quality vs. fuel-efficiency trade-offs between low-end and high-end car models. I first consolidated a cross-sectional dataset of 10,000+ observations (2005–2014) and 22 variables from 3 sources. I then developed a Double-Log Regression model to estimate the average miles-per-gallon of an automobile model based on its design features and real market price. 
For feature engineering, I conducted Pearson’s correlation test to detect and reduce multi-collinearity problem; used year-fixed effects to avoid serial correlation. 
<br>
<br>
Models: Linear Regression, Lin-Log and Double-Log models.<br>
Results: Final R_squared: 0.7984 | Final MSE: 0.0024.
<br>
</div>
<center><img src="images/Car_model_corrplot.png"/></center> 

---
## Natural Language Processing

[**Sentiment Analysis on Movie Reviews: Logistic Regression vs. Naive Bayes Bernoulli**](https://github.com/Emmyphung/Sentiment-Analysis)

[![Models](https://img.shields.io/badge/Jupyter-Models-blue?logo=Jupyter)](https://github.com/Emmyphung/Sentiment-Analysis/blob/master/Sentiment%20Analysis%20-%20NLP%20and%20Logistic%20Regression.ipynb)

<div> This notebook will compare the performance of two NLP techniques, Count Vectorizor and TF-IDF Vectorizer, and two classification models, Logistic Regression and Bernoulli Naive Bayes in sentiment analysis. I'll give detailed explanation on which model performs better and why.
</div>
<center><img src="images/Sentiment_analysis.png"/></center> 
<center><img src="images/Sentiment_analysis_math3.png"/></center> 
<br>

---
[**Kaggle Competition: Google QUEST Q&A Labelling**](https://github.com/JasonZhangzy1757/Kaggle_Google_QUEST_QA_Labeling)

[![Models](https://img.shields.io/badge/Jupyter-Models-blue?logo=Jupyter)](https://github.com/JasonZhangzy1757/Kaggle_Google_QUEST_QA_Labeling/blob/master/200128_bert-tf2_treat_question_type_spelling_Cara.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/JasonZhangzy1757/Kaggle_Google_QUEST_QA_Labeling)


<div style="text-align: justify"> Google Q&A Labelling is a classification problem related to NLP. Given pairs of questions and answers, we are asked to classify the question types, answer types, level of helpfulness of the answers, etc. For this project, I conducted comprehensive EDA to understand the datasets and important variables, split the dataset and trained the model individually to solve class imbalance issue, and used BERT pretuned models to process natural language and Tensorflow to solve the classification problem.
</div>
<br>
Models: BERT pretuned model, deep learning model with Tensorflow
</div>
<center><img src="images/Google_Quest_QA.png"/></center>
<br>

---
## Math Modelling

[**Projected growth of Neurendocrine cells using Matlab**](https://github.com/Emmyphung/Neurendocrine-cells)

[![Open Research Paper](https://img.shields.io/badge/PDF-Open_Research_Paper-blue?logo=adobe-acrobat-reader&logoColor=white)](https://github.com/Emmyphung/Neurendocrine-cells/blob/master/Project%20Write-up_My%20Phung.pdf)

<div style="text-align: justify">The project aims at tracking the three phase transformation of neuroendocrine cells specific to the human colon. A stem cell transforms into a progenitor cell and finally a mature cell through symmetric and asymmetric cell division. Symmetric cell division, also known as self-renewal, occurs when a stem cell divides symmetrically into two identical stem cells. Asymmetric cell division characterizes the maturation process when a stem cell divides into a stem cell and a progenitor cell, or a progenitor cell divides into a progenitor cell and a mature cell. In each phase, cells also experience apoptosis. meaning cell death. With an aim to capture this phenomenon, I want to build a model that track the number of cells in each phase, stem cells, progenitor cells and mature cells.
</div>
<center><img src="images/Neucell.png"/></center>

---
## Community Projects

[**Data Science in Brief (@DSinbrief Facebook Page & Group)**](https://www.facebook.com/DSinbrief/)

[![Check out my page](https://img.shields.io/badge/Facebook-View_My_Page-blue?logo=facebook)](https://www.facebook.com/DSinbrief/)

<div style="text-align: justify"> Data Science in Brief (@DSinbrief) is an organization, a learning community that aims at 1) sharing knowledge to inspire young learners and to keep experienced scientists updated with state-of-the-art practices & applications; and 2) connecting young Data Science enthusiasts – learners – practitioners with leading experts in the field for learning and career opportunities
<br>
• Reached 11,000 readers within the 1st month (organically)<br>
• Organized Data Science in Brief: Hands-on Experience & Career Navigation (VN, 2019), a full-day conference that welcomed 146 attendees and 10 guest speakers from giant tech companies (Google, Hitachi, FPT Worldwide) and top universities (Johns Hopkins, NYU, etc.)
</div>
<center><img src="images/DSinbrief_event.png"/></center>
 
---

<center>© 2020 Kunru Lu. Powered by Jekyll and the Minimal Theme.</center>
