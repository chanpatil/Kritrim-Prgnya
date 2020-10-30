# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 18:55:04 2019

@author: Chanabasagoudap
"""
import pandas as pd
import json
import numpy as np
from flask import jsonify 
from flask import Flask, render_template, request,url_for, flash, redirect, jsonify
from werkzeug import secure_filename
from flask_bootstrap import Bootstrap
from itertools import chain, combinations
import matplotlib.pyplot as plt
import base64
import re 
from textblob import TextBlob   
import spacy
import pandas as pd
import os
from nltk.corpus import stopwords
import nltk
import itertools
from pandas_profiling import ProfileReport
#import textblob
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk.probability
from nltk.probability import FreqDist

# creates a Flask application, named app
app = Flask(__name__)
Bootstrap(app)
bootstrapLink= '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/css/bootstrap.min.css">'

# a route where we will display a welcome message via an HTML template

"""Create a list of common words to remove"""
stop_words = stopwords.words('english')

"""Load the pre-trained NLP model in spacy"""
nlp =spacy.load('en_core_web_sm')

# Plotting a graph for Sentiment Analysis
def plot_graph(sentiment_val,file_name,title):
    labels = 'Happy/Positive', 'Concerned/Negative', 'Neutral'
    colors = ['gold', 'red', 'blue']
    explode = (0.1, 0, 0)  # explode 1st slice
    plt.title(title)
    plt.pie(sentiment_val, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    addrs = "results/graphs/"+file_name+".jpg"
    plt.savefig(addrs)
    plt.show()
    return addrs


# Performing Sentiment Analysis
def sentiment_analyser(col):
    #fetching only comment column
    feedb2= col #food, ac
    
    #counting na values
    feedb2.isna().sum()
    
    #removing na values
    feedb2= feedb2.dropna()
    
    #combine row values of the column to one string
    feedb3= ' '.join(map(str, feedb2))
    
    
    #pass input string to TextBlob function
    f1= TextBlob(feedb3)
    
    #polarity means 1 for positive and -1 for negative sentiment
    polarity= f1.sentiment.polarity
    
    #subjectivity means 1 for personal opinion, judgement and emotion
    subjectivity= f1.sentiment.subjectivity
    
    #passing string to textblob function to analyze its sentiment polarity
    f1= TextBlob(feedb3, analyzer= NaiveBayesAnalyzer())
    
    #broad classification if the sentiment is positive or negative
    classification= f1.sentiment.classification
    
    #tokenize the sentence/text input
    tok1= word_tokenize(feedb3)
    tok1= [w.lower() for w in tok1]
    token_word= [w for w in tok1 if w.isalpha()] 
    #print(token_word)
    
    #removing stopwods like is, are etc
    stopw= set(stopwords.words('english'))
    word_no_stop= [w for w in token_word if not w in stopw]
    #print(word_no_stop)
    
    #remove all such respnses that show a neutral sentiment
    drop_words=["na", "not applicable", "nill", "none"]
    
    word_no_stop= [w for w in word_no_stop if not w in drop_words]    
    
    #lemmatize the tokens
    lemmat= WordNetLemmatizer()
    lem_words= [lemmat.lemmatize(x) for x in word_no_stop]
    #print(lem_words)
    #lem_words
    
    #remove duplicates
    def remove(lem_words):
        words= []
        for w in lem_words:
            if w not in words:
                words.append(w)
        return words
    feedb_words= remove(lem_words) 
    
    
    #create a function from VADER to pass all the lemmatize words to it
    sia= SentimentIntensityAnalyzer()
    #create lists of positive, neutral and negative words
    pos_word=[]
    neg_word=[]
    neu_word=[]
    
    #pass the lemmatize words to the function and separate them using compound scores
    for word in feedb_words:
        if (sia.polarity_scores(word)['compound']) >= 0.05:
            pos_word.append(word)
        elif (sia.polarity_scores(word)['compound']) <= -0.05: 
            neg_word.append(word)
        else:
            neu_word.append(word)    
    return sia.polarity_scores(feedb3)



# Plotting Bar Chart
def plot_bar_chart(count_index,location, location_cnt,XLabel,YLabel,title):
    plt.bar(count_index, location_cnt)
    plt.xlabel(XLabel, fontsize=15)
    plt.ylabel(YLabel, fontsize=15)
    plt.xticks(count_index, location, fontsize=9, rotation=0)
    plt.title(title)
    addrs = "results/"+title+".jpg"
    plt.savefig(addrs)
    plt.show()
    return addrs


# Extracting the base64 from image
def extract_base64(plot_img_addr):
    with open(plot_img_addr, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        #print("Encoded String",type(encoded_string))
        encoded_string = encoded_string.decode("utf-8")
        return encoded_string


# Sentiment Analysis
# Sentence Preprocessnig for Sentiment Analysis
def clean_text(text): 
        ''' 
        Utility function to clean tweet text by removing links, special characters 
        using simple regex statements. 
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", text).split()) 


# Extract the aspects of sentence
def get_aspects(sentence):
    doc=nlp(sentence) ## Tokenize and extract grammatical components
    doc=[i.text for i in doc if i.text not in stop_words and i.pos_=="NOUN"] ## Remove common words and retain only nouns
    doc=list(map(lambda i: i.lower(),doc)) ## Normalize text to lower case
    doc=pd.Series(doc)
    aspects = doc.value_counts().head().index.tolist() ## Get 5 most frequent nouns
    return aspects
    
# Extract the sentiment of sentence
def get_text_sentiment(sentence):
    # create TextBlob object of passed tweet text 
        analysis = TextBlob(clean_text(sentence)) 
        # set sentiment 
        if analysis.sentiment.polarity > 0: 
            return 'positive'
        elif analysis.sentiment.polarity == 0: 
            return 'neutral'
        else: 
            return 'negative'


##############################################################################################################
@app.route("/")
def hello():
    return render_template('index.html')

@app.route('/dataset_analyse', methods = ['GET', 'POST'])
def dataset_analyse():
    if request.method == 'POST':
        dataset = pd.read_excel(request.files.get('file'),encoding='utf-8', error_bad_lines=False)
        
        if dataset.shape is None:
            return "Uploaded file is Empty"
        else:
            #print("Heelllo")
            result = { 
                    "Shape of Dataset" : list(dataset.shape),
                    "Columns in Dataset" : list(dataset.columns),
                    "Count of Columns" : str(len(dataset.columns)),
                    "Need DataPreprocessing w.r.t NaN" : str(dataset.isnull().values.any()),
                    "Total Null Value count" : str(dataset.isnull().sum().sum()),
                    "Column with Null Values" : list(dataset.columns[dataset.isnull().any()])}
            #print("Result is ",result)
            #return json.dumps(result)
            return bootstrapLink+"<div class='container container-fluid'><div class='panel panel-default'><div class='panel-heading text-center'><h3 class='panel-title'>Dataset Analysis</h3></div><div class='panel-body'><div class='row'><div class='col-md-12'><table class='table table-bordered table-hover'><tr><th>Shape of Dataset(rows/columns)</th><th>Columns in Dataset</th><th>Count of Columns</th><th>Need DataPreprocessing w.r.t NaN</th><th>Total Null Value count</th><th>Column with Null Values</th></tr><tr><td>"+str(dataset.shape)+"</td><td>"+str(dataset.columns)+"</td><td>"+str(len(dataset.columns))+"</td><td>"+str(dataset.isnull().values.any())+"</td><td>"+str(dataset.isnull().sum().sum())+"</td><td>"+str(dataset.columns[dataset.isnull().any()])+"</td></tr></table></div></div></div></div></div>"                                     
            #return "<table border='1'><tr><th>Shape of Dataset</th></tr><tr><td>"+str(len(dataset.columns))+"</td></tr></table>"                                     

@app.route('/detailed_analysis', methods = ['GET', 'POST'])
def detailed_analysis():
    if request.method == 'POST':
        missing_values = ["n/a", "na", "--","\n","NaN"]
        dataset = pd.read_excel(request.files.get('file'),encoding='utf-8', error_bad_lines=False, na_values=missing_values)
        if dataset.shape is None:
            return "Uploaded file is Empty"
        else:
            numerical_data = dataset._get_numeric_data().columns
            categorical_data = [c for i, c in enumerate(dataset.columns) if dataset.dtypes[i] in [np.object]]
            Nan_Values = dataset.isnull().sum()

            primary_keys = [] 
            
            # Dynamically generating the possible Primary Keys
            for feature in dataset.columns:
                if len(dataset[feature]) == dataset[feature].nunique():
                    primary_keys.append(feature)
            #print("Primary Key",primary_keys)
            
            # Dynamically generating the count of primary keys
            pk_count_list = []
            for feature in primary_keys:
                pk_count_list.append(len(dataset[feature].unique()))
            
            # Column Analysis
            column_analysis = {}
            for features in dataset.columns:
                column_analysis[features] = dataset[features].describe()
            #print("Column Analysis",column_analysis)
            
            result = bootstrapLink+"<div class='container container-fluid'><div class='panel panel-default'><div class='panel-heading text-center'><h3 class='panel-title'> Types of Descriptive Analysis</h3></div><div class='panel-body'><div class='row'><div class='col-md-12'><table class='table table-bordered table-hover'><tr><th>Column with Numeric Values</th><th>Categorical Data</th><th>NaN values Columns</th><th>Primary Keys</th><th>Primary Keys Count List</th><th>Columns Analysis</th></tr><tr><td>"+str(numerical_data)+"</td><td>"+str(categorical_data)+"</td><td>"+str(Nan_Values)+"</td><td>"+str(primary_keys)+"</td><td>"+str(pk_count_list)+"</td><td>"+str(column_analysis)+"</td></tr></table></div></div></div></div></div>"
            return result

@app.route('/exploratory_data_analysis', methods = ['GET', 'POST'])
def exploratory_data_analysis():
    if request.method == 'POST':
        dataset = pd.read_excel(request.files.get('file'),encoding='utf-8', error_bad_lines=False)
        
        if dataset.shape is None:
            return "Uploaded file is Empty"
        else:            
                       
            # Visulization 1 ---- Feedback from each office Location
            #tot_feedback_count = len(dataset.Branch)
            viz_cnt,hitech_cnt, bachu_cnt, knodapur_cnt, kolhaput_cnt= 0, 0, 0, 0, 0
            for location in dataset.Branch:
                if location == "Visakhapatnam":
                    viz_cnt += 1
                if location == "Hitech City":
                    hitech_cnt += 1
                if location == "Bachupally":
                    bachu_cnt += 1
                if location == "Kondapur":
                    knodapur_cnt += 1
                if location == "Kolhapur":
                    kolhaput_cnt += 1
            
            location = ["Visakhapatnam","Hitech","Bachupally","Kondapur","Kolhapur"]
            count_index = np.arange(len(location)) 
            location_cnt = [viz_cnt,hitech_cnt, bachu_cnt,knodapur_cnt, kolhaput_cnt]
            XLabel1 = "MOURIT Tech Office Location"
            YLabel1 = "Feedback from different Office Location"
            title1 = "Feedback from Different Office Loaction of MOURI Tech"
            plot_img_addr1 = plot_bar_chart(count_index,location, location_cnt,XLabel1,YLabel1,title1)

            # Calling function to extact the base64 fror title1
            encoded_string1 = extract_base64(plot_img_addr1)
                     
            div1 = "<div style='font-size: 20px; font-weight: bold; color: blue; text-align: center; border-style: solid; border-color: black;'><p>"+title1+"<br><img src='data:image/jpeg;base64,"+encoded_string1+"'/></div>"
            
           # Visualization 2 :------  Internet Performacnce
           
            internet_sat , internet_unsat , internet_NA = 0, 0 ,0
            for emotion in dataset["internet performance"]:
                if emotion == "Yes":
                    internet_sat += 1
                if emotion == "No":
                    internet_unsat += 1
                if emotion == "N/A":
                    internet_NA += 1
            
            emotion = ["Satisfied", "Unsatisfied"," Not Applicable"]
            emotion_cnt = np.arange(len(emotion))
            internet_cnt = [internet_sat,internet_unsat, internet_NA]
            XLabel2 = " Happiness index"
            YLabel2 = " Internet Performance of different Office Location"
            title2 = "Overall MOURI Tech internet Performance"
            plot_img_addr2 = plot_bar_chart(emotion_cnt,emotion, internet_cnt,XLabel2,YLabel2,title2)
            
            # Calling function to extact the base64 fror title2
            encoded_string2 = extract_base64(plot_img_addr2)
            
            div2 = "<div style='font-size: 20px; font-weight: bold; color: blue; text-align: center; border-style: solid; border-color: black;'><p>"+title2+"<br><img src='data:image/jpeg;base64,"+encoded_string2+"'/></div>"


            # Visualization 3 :----------------------- Skype Zoom and WebEX connectivity
            
            szw_sat ,szw_unsat ,skw_NA = 0, 0, 0
            
            for connectivity in dataset["Skype/Zoom/Webex_Call_stabality"]:
                if connectivity =="Yes":
                    szw_sat += 1
                if connectivity == "No":
                    szw_unsat +=1
                if connectivity =="N/A":
                    skw_NA += 1
            
            szw_cnt = [szw_sat, szw_unsat,skw_NA ]
            XLabel3 = " Happiness index"
            YLabel3 = " Skype Zoom and WebEX connectivity different Office Location"
            title3 = "Overall MOURI Tech Skype Zoom and WebEX connectivity  Performance"
            
            #Plotting and saving a graph
            plot_img_addr3 = plot_bar_chart(emotion_cnt,emotion, szw_cnt,XLabel3,YLabel3,title3)
            # Calling function to extact the base64 fror title3
            encoded_string3 = extract_base64(plot_img_addr3)
            div3 = "<div style='font-size: 20px; font-weight: bold; color: blue; text-align: center; border-style: solid; border-color: black;'><p>"+title3+"<br><img src='data:image/jpeg;base64,"+encoded_string3+"'/></div>"
            
            return ""+div1+"<br>"+div2+"<br>"+div3+""
         
        
@app.route('/data_profiling', methods = ['GET', 'POST'])
def data_profiling():     
    datasetprofile = pd.read_excel(request.files.get('file'),encoding='utf-8', error_bad_lines=False)
    #profile = datasetprofile.profile_report(title='Data Profiling Report')
    #profile.to_file(output_file="DFReport.html")
    return render_template('DFReport.html')
    
@app.route('/aspect_analysis', methods = ['GET', 'POST'])
def aspect_analysis():           
    if request.method == 'POST':
        missing_values = ["n/a", "na", "--","\n","NaN","nan","nill","No","No Comments"]
        dataset = pd.read_excel(request.files.get('file'),encoding='utf-8', error_bad_lines=False, na_values=missing_values)
        if dataset.shape is None:
            return "Uploaded file is Empty"
        else:
            feedback = dataset["Suggestion"]
            #print(len(feedback))
            
            aspectlist = []
            
            proc_feedback = []
            for sentence in feedback:
                 # Function to extract the feedback
                 if (isinstance(sentence, str)):
                     proc_feedback.append(nltk.sent_tokenize(sentence))
            for suggestion in proc_feedback:
                
                aspects = get_aspects(str(suggestion))
                sentiments = get_text_sentiment(str(suggestion))
                     
                 # Function to extract the feedback sentiment
                if aspects not in aspectlist:
                    aspectlist.append(aspects)
        
            #print("Aspect List", aspectlist) 
            #print("Sentiments",sentiments)
            #print("Proceeessed Feedbacl",proc_feedback)
            
            final_aspects = list(itertools.chain(*aspectlist))
            aspect_count = {i:final_aspects.count(i) for i in final_aspects}
            final_aspectz = list(dict.fromkeys(final_aspects))
            #print("Aspect Count",aspect_count)
            aspect_count = sorted(aspect_count.items(), key = lambda kv:(kv[1], kv[0]))
            #print("Final Aspect List", final_aspectz) 
            IT_aspects_tuple=np.array(aspect_count)
            IT_aspects = IT_aspects_tuple.T[0]
            IT_aspects_cnt = IT_aspects_tuple.T[1]
            
            IT_df = {'Aspects':IT_aspects,'Aspects Count':IT_aspects_cnt}
            IT_aspects_counts_df = pd.DataFrame(IT_df)
            #print(IT_aspects_counts_df)
            # Visualization4: -------- Visualizing the most spoke topic
            
            XLabel4 = " Mostly emplyee are talking about these Aspects "
            YLabel4 = " Aspect Count"
            title4 = "Employee Issue Versus IssueOccurance "
            IT_aspects =IT_aspects[-9:]
            IT_aspects_cnt = IT_aspects_cnt[-9:]
            IT_aspects_count = np.arange(len(IT_aspects))
            plot_img_addr4 = plot_bar_chart(IT_aspects_count,IT_aspects, IT_aspects_cnt,XLabel4,YLabel4,title4)
            
            # Calling function to extact the base64 fror title2
            encoded_string4 = extract_base64(plot_img_addr4)
            
            #Sentiment Analysis Code
            
            fianlized_aspects = IT_aspects[-9:]
            for suggestion in proc_feedback:
                
                aspects = get_aspects(str(suggestion))
                sentiments = get_text_sentiment(str(suggestion))
                     
                 # Function to extract the feedback sentiment
            internet_pos, intenrnet_neg,internet_neu = 0, 0, 0
            calls_pos, calls_neg, calls_neu = 0, 0 , 0
            connectivity_pos , connectivity_neg,connectivity_neu = 0, 0, 0
            sfb_pos, sfb_neg, sfb_neu = 0, 0, 0
            sentimental_aspects = ["internet","calls","connectivity","buisiness"]
            
            for suggestion in proc_feedback:
                aspects = get_aspects(str(suggestion))
                sentiments = get_text_sentiment(str(suggestion))
                if aspects in sentimental_aspects:
                    if aspects == "internet":
                        if sentiments =="positive":
                            internet_pos += 1
                        elif sentiments == "negative":
                            intenrnet_neg += 1
                        elif sentiments =="neutral":
                            internet_neu += 1
                    elif aspects =="calls":
                        if sentiments =="positive":
                            calls_pos += 1
                        elif sentiments == "negative":
                            calls_neg += 1
                        elif sentiments =="neutral":
                            calls_neu += 1
                    elif aspects =="connectivity":
                        if sentiments =="positive":
                            connectivity_pos += 1
                        elif sentiments == "negative":
                            connectivity_neg += 1
                        elif sentiments =="neutral":
                            connectivity_neu += 1
                    elif aspects == "buisiness":
                        if sentiments =="positive":
                            sfb_pos += 1
                        elif sentiments == "negative":
                            sfb_neg += 1
                        elif sentiments =="neutral":
                            sfb_neu += 1
            
            internet_count = [internet_pos,intenrnet_neg, internet_neu]
            calls_count = [calls_pos,calls_neg, calls_neu]
            connectivity_count = [connectivity_pos,connectivity_neg, connectivity_neu]
            buisiness_count = [sfb_pos, sfb_neg,sfb_neu]
            #print("Internet Count", internet_count)
            #print("\n")
            #print("calls_count",calls_count)
            #print("\n")
            #print("connectivity_count",connectivity_count)
            #print("buisiness_count",buisiness_count)
            
            
            div1 = "<table border='1'><tr><th>Most famous Aspect Employee Talk About</th><tr><td>"+str(aspect_count)+"</td></tr></table>"
            div2 = "<div style='font-size: 20px; font-weight: bold; color: blue; text-align: center; border-style: solid; border-color: black;'><p>"+title4+"<br><img src='data:image/jpeg;base64,"+encoded_string4+"'/></div>"
            return ""+div1+"<br>"+div2+""
            
@app.route('/sentiment_analytics', methods = ['GET', 'POST'])
def sentiment_analytics():
    if request.method == 'POST':
        feedb1 = pd.read_excel(request.files.get('file'),encoding='utf-8', error_bad_lines=False)
        
        if feedb1.shape is None:
            return "Uploaded file is Empty"
        else:
            del_list = ["Start time","Completion time","Email","Name","Branch"]
            result_list = set(feedb1.columns)-set(del_list)

            sentiment = {}
            for col in result_list:
                col_val = feedb1[col]
                result = sentiment_analyser(col_val)
                #print(result)
                sentiment[col] = result
            
            #print("Sentiment is as follows",sentiment)
            
            #pi_plot("Service Request",srr_vals*100,aspect_happy_index,colors)
            
            #print(int(100 * sentiment["VPN connectivity"]["neg"]))
            sent_sugg = [int(100 * sentiment["Suggestion"]["pos"]), int(100 * sentiment["Suggestion"]["neg"]),int(100 * sentiment["Suggestion"]["neu"])]
            sent_VPN_conn = [int(100 * sentiment["VPN connectivity"]["pos"]), int(100 * sentiment["VPN connectivity"]["neg"]),int(100 * sentiment["VPN connectivity"]["neu"])]
            sent_SZW = [int(100 * sentiment["Skype/Zoom/Webex_Call_stabality"]["pos"]),int(100 * sentiment["Skype/Zoom/Webex_Call_stabality"]["neg"]),int(100 * sentiment["Skype/Zoom/Webex_Call_stabality"]["neu"])]
            sent_SRR = [int(100 * sentiment["Service request resolutions"]["pos"]), int(100 * sentiment["Service request resolutions"]["neg"]),int(100 * sentiment["Service request resolutions"]["neu"])]
            sent_internet_per = [int(100 * sentiment["internet performance"]["pos"]), int(100 * sentiment["internet performance"]["neg"]),int(100 * sentiment["internet performance"]["neu"])]
            sent_voip_qual = [int(100 * sentiment["VOIP(Phone) quality"]["pos"]), int(100 * sentiment["VOIP(Phone) quality"]["neg"]),int(100 * sentiment["VOIP(Phone) quality"]["neu"])]
            
            sent_suggestion_addr1 = plot_graph(sent_sugg,"Overall Suggestion","Setiment Analysis for Overall suggestion")
            sent_VPN_addr2 =plot_graph(sent_VPN_conn,"VPN connectivity","Sentiment Analysis for VPN Connectivity")
            sent_SZW_img_addr3 = plot_graph(sent_SZW,"Skype_Zoom_WebEX","Sentiment Analysis for Skype Zoom and WebEx call")
            sent_SR_addr4 =plot_graph(sent_SRR,"ServiceRequest","Sentiment Analysis for Service Request Resolution")
            sent_Internet_addr5 =plot_graph(sent_internet_per,"InternetPerformance","Sentiment Analysis for Internet Performance")
            sent_VOIP_addr6 = plot_graph(sent_voip_qual,"VOIP Quality","Internet Performance for VOIP Quality")
        
            encoded_string1 = extract_base64(sent_suggestion_addr1)
            encoded_string2 = extract_base64(sent_VPN_addr2)
            encoded_string3 = extract_base64(sent_SZW_img_addr3)
            encoded_string4 = extract_base64(sent_SR_addr4)
            encoded_string5 = extract_base64(sent_Internet_addr5)
            encoded_string6 = extract_base64(sent_VOIP_addr6)

            div1 = "<div style='font-size: 20px; font-weight: bold; color: blue; text-align: center; border-style: solid; border-color: black;'><img src='data:image/jpeg;base64,"+encoded_string1+"'/></div>"    
            div2 = "<div style='font-size: 20px; font-weight: bold; color: blue; text-align: center; border-style: solid; border-color: black;'><img src='data:image/jpeg;base64,"+encoded_string2+"'/></div>"    
            div3 = "<div style='font-size: 20px; font-weight: bold; color: blue; text-align: center; border-style: solid; border-color: black;'><img src='data:image/jpeg;base64,"+encoded_string3+"'/></div>"    
            div4 = "<div style='font-size: 20px; font-weight: bold; color: blue; text-align: center; border-style: solid; border-color: black;'><img src='data:image/jpeg;base64,"+encoded_string4+"'/></div>"    
            div5 = "<div style='font-size: 20px; font-weight: bold; color: blue; text-align: center; border-style: solid; border-color: black;'><img src='data:image/jpeg;base64,"+encoded_string5+"'/></div>"    
            div6 = "<div style='font-size: 20px; font-weight: bold; color: blue; text-align: center; border-style: solid; border-color: black;'><img src='data:image/jpeg;base64,"+encoded_string6+"'/></div>"
           
            div = div2+div3+div4+div5+div6+div1

            
    return bootstrapLink +"<div class='container container-fluid'><div class='panel panel-default'><div class='panel-heading text-center'><h3 class='panel-title'><strong>Sentiment Analysis</strong></h3></div><br>" + div

"""
@app.route('/predictive_analytics_renderer', methods = ['GET', 'POST'])
def predictive_analytics_renderer():
    
    return render_template('predict.html')
            

@app.route('/predict_internet', methods = ['GET', 'POST'])
def predict_internet():
    
    return "work in progress"

@app.route('/predict_SRR', methods = ['GET', 'POST'])
def predict_SRR():
    
    return "work in progress"


@app.route('/predict_call_conn', methods = ['GET', 'POST'])
def predict_call_conn():
    
    return "work in progress"

@app.route('/predict_VPN', methods = ['GET', 'POST'])
def predict_VPN():
    
    return "work in progress"

@app.route('/predict_VOIP', methods = ['GET', 'POST'])
def predict_VOIP():
    
    return "work in progress"

"""











            
# run the application
if __name__ == "__main__":
    app.run()