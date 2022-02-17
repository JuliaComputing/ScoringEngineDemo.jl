# End-to-End machine-learning workflow on JuliaHub

> Welcome to this webinar covering an end-to-end machine learning workflow. Today, we’ll cover how a complete machine-learning can be built using Julia. We'll leverage JuliaHub, yet all the code is compatible to run locally on a regular laptop. Key steps from data ingestion to deployment as a web service will be covered, including preprocessing and hyper-parameter search. We will pay special attention on building a reproducible process and taking advantage of distributed computing to accelerate model discovery. 

## Intro

The problem we'll tackle today come from Insurance Pricing Challenge hosted by AI Crowd. 
For today's task, we'll develop a model to assess the probability that an insured file a claim during the next year.

This takes the form of a the very common binary regression problem. It is similar to scenarios where one wants to assess an applicant's credit risk, or the probability to convert a prospect into a sale or renew a contract. 

All the code is available as a Julia project on JuliaComputing's GitHub.

In the upcoming steps, we'll look to bring this project to deployment, hosting web api serving our model predictions, both for a simple neural network and a gradient boosted trees.

Let's go ahead and start with the data ingestion.

## Data Loading

First step is obviously to load our data. In this case, the csv file has been already downloaded and placed into a assets folder. 
Julia CSV package provides one of the most performance csv parser around, making it convenient to work directly from such format, even for large files. 

Should take less than 1/3 of a sec to load this 228,216 x 26 data. 

Julia depending where data lives, Julia has multiple connectors, whether it is for Arrow or Parquet files, connections to SQL DB like SQLite or Postgres or cloud storage. 
JuliaHub also has its own DataSets management system, facilitating their versioning. 

## Data Preprocessing

A critical step priori to jumping into modeling is to get the data cleaned and formatted to be compatible with the desired alogirthms. 
This would typically be done hand on hand with a data exploratory analysis, or EDA. 

Data visualisation can be handy at this stage. We'll keep this topic for an upcoming session where we'll 


## Model  Training

### Extract hyper-search distributed results

```
tar -zxvf assets\\results.tar -C assets\\
```
