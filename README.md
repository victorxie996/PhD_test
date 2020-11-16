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
7. Download the pretrained seq2seq and bert models from [here](https://drive.google.com/file/d/1mw0VENGVosXo0yct7KRXxq6LPNPVJDNT/view?usp=sharing) and [here](https://drive.google.com/file/d/1M5Go5VM-fsXpvYfPxMH1vvnphVa4hHBu/view?usp=sharing). Then ppoy the trained models (brain and pth files) to the correct path:
```
cp <your brain file path> ./application/model/model.brain 
AND
cp <your brain file path> ./application/model/epoch_1.pth 
```
8. run Flask server in virtual environment
```
bash start.sh
OR
./start.sh
```

