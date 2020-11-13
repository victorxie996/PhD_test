# PhD_test
My PhD programming test

# Introduction
We use a seq2seq abstractive text summarisation model based on GRU encoder decoder with attention mechanism.

# Base Info

Data: News summary dataset obtained from [Kaggle](https://www.kaggle.com/sunnysai12345/news-summary?select=news_summary_more.csv)


# Requirements
```
Python 3
```

```
Pip
```


# Installation
1. Download venv and setup virtual envirement
```
pip3 install virtualenv
```
2. Clone this repository
```
git clone https://github.com/victorxie996/PhD_test.git
```
3.  Change directory to the repository, e.g.:
```
cd PhD_test
```
4. Create, activate and get into the virtual environment 
```
pip3 venv env. env\bin\activate

or 

.\env\Scripts\activate
```
5. Install required library
```
pip3 install -r requirements.txt
```
6. Install spacy english model
```
python3 -m spacy download en
```
7. run Flask server in virtual environment
```
bash start.sh
```



