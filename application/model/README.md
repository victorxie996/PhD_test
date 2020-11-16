# Compendium
## Introduction
Compendium is a seq2seq abstractive text symmetrization model based on GRU encoder decoder with attention mechanism.  
## Base Info
Files and there uses are listed below
     
    Data -> Folder used for train and test data storage
    helper -> Contains helper functions used in model.ipynb.
    brain -> Contains trained RNN Model.
    Data Clean.ipnb -> Ipython Notbook for cleaning and splitting data into train, val and test.
    model.ipynb -> Ipython Notebook for training andd testing model.
    requirement -> txt file containg required lib
    
## Requirements

`Python : 3.X`

`Pip`

## Installation
Steps for installation?<br>
1. Download venv<br>
    `pip3 install virtualenv`
2. Clone<br>
    `git clone https://github.com/soni-ratnesh/compendium.git`
3. Change directory<br>
    `cd compendium                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      `
4. Create and activate virtual environment<br>
    `pip3 venv env`<br> 
    `. env\bin\activate`
5. Install required library<br>
    `pip3 install -r requirements.txt`  
6. Install spacy english model<br>
    `python3 -m spacy download en`<br>
7. Copy brain file<br>
    `cp <your barin file path> ./application/model/brain`<br>
8. Run Flask server<br>
    `bash start.sh`

## Result
The testing accuracy and loss are,<br>

    Test Loss     :  2.23
    Test PPE      :  10.87

## Need trained model?
We do provide trained model, just let us know. you can get trained model from [here](https://drive.google.com/file/d/1v4Ja_5NAHfUe4e_cJi1wGFI8XojVTyfV/view?usp=sharing) . Save the provideed file in brain dir to load and run.


## Contribution
Pull requests are welcome. If have an idea please let me know through an issue.
For contribution please raise pull requests by,


 1. Clone Repo

    `git clone https://github.com/soni-ratnesh/compendium.git`

 2. Install Dependencies
 
    `pip3 install requirements.txt`

 3. Verify your changes 

 4. [Submit Pull Request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request)

## Short for time??<br>                                                  
Feel free to raise an [issue](https://github.com/soni-ratnesh/compendium/issues) to correct errors or contribute content without a pull request.
