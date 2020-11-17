PhD_test
====
My PhD programming test

Introduction
-------
The project was built using Pytorch, Spacy and torchtext. Pytorch is used to build the summarization model, while Spacy and torchtext was used to pre-process data. We use a seq2seq abstractive text summarisation model based on GRU encoder decoder with attention mechanism. To identify the authenticity of the news, we use the fine-tuned pretrained BERT model.

In the 'Notebook' foler, it has the training and testing source code for both seq2seq and fine-tuned BERT models. In the 'Application' folder, we only use the test files and the pretrained models to run the App.

Base Info
-------
Data: News summary dataset obtained from [Kaggle](https://www.kaggle.com/sunnysai12345/news-summary?select=news_summary_more.csv); Fake and real news dataset obtained from [Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset).


Requirements
-------
```
Python 3
```

```
Pip
```


Installation
-------
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

Usage
-------
After get starting the server, the server looks like: 

![image](https://github.com/victorxie996/PhD_test/blob/main/demo/bug_img.png)

Next, type the address on a broswer: http://127.0.0.1:5000/?text= ```the news you would like to feed into the model```, press ENTER and the server will return text summary and prediction of whether the it is a true news.

Demonstration
-------
Lets say we want to input a news:

text= Several Bengaluru civic workers protested in front of the Bruhat Bengaluru Mahanagara Palike Office on Tuesday alleging that the contractors haven't paid them since three months. They complained that the contractors haven't been marking their attendance and weren't providing them with the safety and cleaning equipment. They further alleged they aren't given an off even when they are sick.


The output of the model looks like: 

![image](https://github.com/victorxie996/PhD_test/blob/main/demo/result_1.png)

I hope you find it useful!:)
