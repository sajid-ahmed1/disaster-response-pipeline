# Disaster Response Pipeline Project

## Motivation
The motivation for this project was an assignment from my Udacity Data Scientist Nanodegree with data supplied from Appen (formerly known as Figure 8 - https://www.figure-eight.com/). The dataset contains two csv files: messages and categories. Message.csv contains the messages from diasater victims and the Categories.csv is the category which the message fits. For example, the message "Is the Hurricane over or is it not over" would fall under "aid-related".

The project is split into three main sections:
1. An ETL pipeline that takes the two csv files mentioned above and creates a new dataframe that merges the two and cleans the data, with finally saving the dataframe in a SQLite database.
2. A machine learning pipeline that cleans the message data to tokenise (to split the sentence into multiple word strings), remove stopwords (words with no meaning but are used to make the sentence readable) and lemanitise (taking words to their root form). After that, there are a few models that are used to train and find the best model and it's parameters.
3. A flask web app that displays the model in a way that disaster response organisations can use to categories messages quickly and efficiently.

## Getting Started

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/disaster_response_model.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Go to http://0.0.0.0:3001/

## Libaries Used

**Data Manipulation and Analysis:**
- Pandas
- numpy
- re

**Web App**
- json
- plotly
- flask

**Machine Learning**
- scikit-learn
- nltk

**System/Data Warehouse**
- sqlalchemy
- joblib

## Files within repository
- run.py: can be used to run the flask web app from your local machine 
- templates folder: contains two html files which store the code used to display the web app
- process_data.py: the ETL pipeline used to load,merge and clean the two datasets
- ETL Pipeline Preparation.ipynb: additional information on the ETL pipeline as a Jupyter Notebook
- ML Pipeline Preparation.ipynb: additional information on the ML pipeline as a Jupyter Notebook
- train_classifier.py: the ML pipeline used to train and evaluate the ML models used to classify the disater dataset

## Screenshots from the Web App
These are screenshots from the Web App ran on a local server.

**Top Section**
A navigation bar at the top helps to direct the user to either the main page, Udacity page for more information, a link to my Github and a link to my Linkedin.
Below that is the section where the user can input their message from the disaster victim and they can submit it by clicking on the "Classify message" button.
<img width="1270" alt="image" src="https://github.com/user-attachments/assets/402347b4-1a50-4096-a35c-e658f7e55eff">

**Classifying Message**
After submitting the message, the model classifies the message and highlights the categories it thinks are related to the message. 
<img width="1270" alt="image" src="https://github.com/user-attachments/assets/00b4f0b6-e79a-4a36-b43b-b354dac5dfa1">

**Graph Section**
There are three graphs that display some information on the dataset. The count of messages based on their genre displayed as a bar chart and a pie chart.
<img width="1270" alt="image" src="https://github.com/user-attachments/assets/d3d1b081-6e16-4c9f-a40e-c60ff2971e6b">

Below is a quick analysis on the imbalance of the categories. From the bar chart, we can see the related category has the highest count of messages around 20k while the next catergory with the most messages is aid related with around 10k. There are some categories with barely any data and is quite diffcult to see them on the chart. The issue this presents is when classifying messages, there could be a heavy bias on the related category as the model has more information on this compared to other underrepresented categories. To combat this imbalance, we can undersample the related cateogory or oversmaple the underrepresented categories. In regards with the evaluation, we can split the categories to three classes: common, a balanced approach between precision and recall will be effective because we have an abundance of training data; critical, we should prioritise recall to highlight more false positives as risks are still potential disasters ; and less critical, we should prioritise precision to avoid false positives.
<img width="1270" alt="image" src="https://github.com/user-attachments/assets/b321efcc-0974-4337-957d-189b84d83401">


## Licensing, Authors, and Acknowledgement
- Data source: Appen (formerly known as Figure 8 - https://www.figure-eight.com/)
- Acknowledgement: Udacity Nanodegree - https://www.udacity.com/
- License: MIT License
  
Copyright (c) [2024] [Sajid Ahmed]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

