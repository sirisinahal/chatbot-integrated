
# In[1]:


#nltk helps the computer to analyze, preprocess and understand the written text.
#intent-question converted to intents

import csv
import nltk
from tqdm import tqdm
# nltk.download()
import os
import random
import pandas as pd
import pickle
import re
from csv import reader
import numpy as np

from datetime import datetime

from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer #used in lemmatization
lemmatizer = WordNetLemmatizer()
from num2words import num2words
from itertools import chain

import enchant
import os

import spacy
from spacy import displacy

import requests
from bs4 import BeautifulSoup

from spacy.language import Language
import wikipedia
from gensim.test.utils import datapath
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
import math
from scispacy.abbreviation import AbbreviationDetector


# ### Abbreviation detection

# In[2]:


class Abbreviations_Detection():
        
    def replace_abb(self,abbreviations,user_query):
        y=user_query.split()
        for j in y:
            l={}
            for k in abbreviations.keys():
                if k.lower()==j or k==j:
                    d=abbreviations[k]
                    for m in d.keys():
                        if m.lower() in user_query:
                            user_query=user_query.replace(m.lower(),d[m])
                        elif m in user_query:
                            user_query=user_query.replace(m,d[m])
        user_query=re.sub(r'[()]',' ',user_query)
        x=user_query.split()
        for k in range(len(x)):
            try:
                if x[k]==x[k+1]:
                    x.remove(x[k+1])
            except:
                pass
        return " ".join(w for w in x)
    
    def Update_abbreviations(self,filename,filename1):
        with open(filename,encoding='utf8') as myFile:  
            reader = csv.DictReader(myFile)
            abbreviations={}
            for row in reader:
                abbreviations[row['full form']]=row['short form']
                
        abbreviations1={}
        for i in abbreviations.keys():
            y=i.split()
            if y[0] not in abbreviations1.keys():
                abbreviations1[y[0]]={i:abbreviations[i]}
            else:
                abbreviations1[y[0]][i]=abbreviations[i]
                
        with open(filename1,encoding='utf-8') as myFile:  
            reader = csv.DictReader(myFile)
            responses=[]
            questions=[]
            for row in reader:
                text=row['responses']
                text1=row['questions']
                responses.append(text)
                questions.append(text1)
                
        for i in range(len(questions)):
            responses[i]=self.replace_abb(abbreviations1,responses[i])
            questions[i]=self.replace_abb(abbreviations1,questions[i])
         
        with open('abbreviated_corpus_s.csv','w',newline='',encoding='utf-8') as file:
            writer=csv.writer(file)
            writer.writerow(['questions','responses'])
            for i in range(len(questions)):
                writer.writerow([questions[i],responses[i]])
        
        
        filename1 = 'abbreviations_s.pickle' 
        outfile = open(filename1,'wb') 
        pickle.dump(abbreviations1,outfile) 
        outfile.close()
        return filename1
    
    def Abbreviations(self,filename):
        nlp=spacy.load('en_core_web_sm')
        def get_abbre_detector(nlp, name):
            return AbbreviationDetector(nlp)
        Language.factory('detector',func=get_abbre_detector)
        nlp.add_pipe('detector',last=True)
        
        with open(filename,encoding='utf-8') as myFile:  
            reader = csv.DictReader(myFile)
            responses=[]
            questions=[]
            for row in reader:
                text=row['responses']
                text1=row['questions']
                responses.append(text)
                questions.append(text1)
        
        responses.extend(questions)
        
        abb2=set()
        for i in range(len(responses)):
            a=re.findall(r'(([A-Z][a-z]+[ ][andof\s]*){3,})',responses[i])
            if a!=[]:
                for j in a:
                    if j[0] not in abb2:
                        t=j[0]
                        t=t[:-1]
                        t=t.replace('The ','')
                        abb2.add(t)

        dis={}
        a=set()
        for i in abb2:
            y=i.split()
            a=''
            b=''
            if 'and' in y or 'of' in y:
                for j in y:
                    b+=j[0].upper()
                    if j not in ['and','of']:
                        a=a+j[0]

            else:
                for j in y:
                    a+=j[0]
            if b!='':
                dis[i]=b
            else:
                pass
            dis[i]=a

        a=set()
        for i in range(len(responses)):
            ab=re.findall(r'[A-Z]{2,}',responses[i]) 
            if ab!=[]:
                for i in ab:
                    a.add(i)

        dis1={}
        for i in dis.keys():
            if dis[i] in a:
                dis1[i]=dis[i]

        def get_abb(a):
            abbreviations={}
            for i in a:
                req=requests.get('https://acronyms.thefreedictionary.com/'+i)
                Soup=BeautifulSoup(req.content,'html.parser')
                body=Soup.find('body')
                table=body.find('table',{'id':'AcrFinder'})
                try:
                    tr=table.findAll('tr')
                    for j in range(1,len(tr)):
                        td=tr[j].findAll('td')
                        j1=td[1].find('i')
                        f=td[1].text
                        if j1==None:
                            pass
                        else:
                            r=j1.text
                            f=f.replace(r,'')
                        if f[-1]==' ':
                            f=f[:-1]
                        for k in range(len(responses)):
                            if f in responses[k]:
                                if f not in abbreviations:
                                    abbreviations[f]=i
                except:
                    pass
            return abbreviations

        left_out=[]
        for i in a:
            if i not in dis1.values():
                left_out.append(i)

        abb1=get_abb(left_out)

        for i in dis1.keys():
            if i not in abb1.keys():
                abb1[i]=dis1[i]
        
        def detector(text): 
            doc=nlp(text)
            abbreviation={}
            for ab in doc._.abbreviations:
                abbreviation[str(ab._.long_form)]=str(ab)
            return abbreviation
        
        def get_from_web(c):
            r =requests.get('https://en.wikipedia.org/wiki/'+c)
            soup=BeautifulSoup(r.content,'html.parser')
            divtag=soup.body.findAll('div',{'class':'mw-parser-output'})
            try:
                ptag=divtag[0].findAll('p')
                info=ptag[0].text
                info=info+ptag[1].text
                info=info+ptag[2].text
            except:
                try:
                    ultag=divtag[0].findAll('ul')
                    info=ultag[0].text
                except:
                    info='not available'
            info=re.sub(r'/.*/|\[.*?\]','',info)
            return info
        
        def fullform_to_abb(sent_tokens,abb):
            text=' '.join(sent for sent in sent_tokens)
            abbreviations={}
            abbreviations=detector(text)

            for i in list(abb):
                di=detector(get_from_web(i))
                for j in di.keys():
                    if j not in abbreviations.keys():
                        abbreviations[j]=di[j]

            left_out=[]
            for i in abb:
                if i not in abbreviations.values():
                    left_out.append(i)

            for i in left_out:
                result=wikipedia.search(i)
                try:
                    if i==result[0]:
                        abbreviations[result[1]]=i
                except:
                    pass
            return abbreviations
        
        abbreviations=fullform_to_abb(responses,a)
        
        for i in abbreviations.keys():
            if i not in abb1.keys():
                abb1[i]=abbreviations[i]
        abbreviations1={}
        for i in abb1.keys():
            y=i.split()
            if y[0] not in abbreviations1.keys():
                abbreviations1[y[0]]={i:abb1[i]}
            else:
                abbreviations1[y[0]][i]=abb1[i]        
        
        for i in range(len(questions)):
            responses[i]=self.replace_abb(abbreviations1,responses[i])
            questions[i]=self.replace_abb(abbreviations1,questions[i])
        
        with open('abbreviated_corpus_s.csv','w',newline='',encoding='utf-8') as file:
            writer=csv.writer(file)
            writer.writerow(['questions','responses'])
            for i in range(len(questions)):
                writer.writerow([questions[i],responses[i]])
        with open('abbreviations_s.csv','w',newline='',encoding='utf-8') as file:
            writer=csv.writer(file)
            writer.writerow(['full form','short form'])
            for i in abbreviations.keys():
                writer.writerow([i,abbreviations[i]])
                
        filename1 = 'abbreviations_s.pickle' 
        outfile = open(filename1,'wb') 
        pickle.dump(abbreviations1,outfile) 
        outfile.close()
        
        return 'abbreviated_corpus_s.csv',filename1,'abbreviations_s.csv'


# In[3]:

    
def abbreviation_check(user_query):
    y=user_query.split()
    for j in y:
        l={}
        for k in abbreviations.keys():
            if k.lower()==j or k==j:
                d=abbreviations[k]
                for m in d.keys():
                    if m.lower() in user_query:
                        user_query=user_query.replace(m.lower(),d[m])
                    elif m in user_query:
                        user_query=user_query.replace(m,d[m])
    user_query=re.sub(r'[()]',' ',user_query)
    x=user_query.split()
    for k in range(len(x)):
        try:
            if x[k]==x[k+1]:
                x.remove(x[k+1])
        except:
            pass
    return " ".join(w for w in x)


# ### Phrases

# In[4]:


class Phrase_Collection():
    
    def Update_phrases(self,filename,filename1):
        with open(filename,encoding='utf-8') as myFile:  
            reader = csv.DictReader(myFile)
            phrases={}
            for row in reader:
                phrases[row['full form']]=row['short form']
                
        phrases1={}
        for i in phrases.keys():
            y=i.split()
            if y[0] not in phrases1.keys():
                phrases1[y[0]]={i:phrases[i]}
            else:
                phrases1[y[0]][i]=phrases[i]
                
        with open(filename1,encoding='utf-8') as myFile:  
            reader = csv.DictReader(myFile)
            responses=[]
            questions=[]
            for row in reader:
                #questlist.append(row['questions'])
                text=row['responses']
                text1=row['questions']
                responses.append(text)
                questions.append(text1)
                
        for i in range(len(questions)):
            questions[i]=self.replace_phrases(phrases1,questions[i])
            responses[i]=self.replace_phrases(phrases1,responses[i])
            

        with open('phrase_corpus_s.csv','w',newline='',encoding='utf-8')as file:
            writer=csv.writer(file)
            writer.writerow(['questions','responses'])
            for i in range(len(questions)):
                writer.writerow([questions[i],responses[i]])
        
        filename1 = 'phrases_s.pickle' 
        outfile = open(filename1,'wb') 
        pickle.dump(phrases1,outfile) 
        outfile.close()
        return filename1
    
    def Phrases_with_hyphen(self,filename):
        with open(filename,encoding='utf-8') as myFile:  
            reader = csv.DictReader(myFile)
            responses=[]
            questions=[]
            for row in reader:
                #questlist.append(row['questions'])
                text=row['responses']
                text1=row['questions']
                responses.append(text)
                questions.append(text1)
        series=[]
        def co_occurences_with_hyphen(sentences):
            # sentences is list of sentences
            co=[]
            for i in range(len(sentences)):
                t=re.findall(r'[A-Za-z0-9]+[-][a-zA-Z0-9]+',sentences[i])
                for j in t:
                    if j not in co:
                        co.append(j)
            sp={}#{chandrayaan:{'chandrayaan-1':'chandrayaan_1'}}
            for i in co:
                y=i.split('-')
                for k in range(10):
                    if str(k) in i:
                        if y[0] not in series:
                            series.append(y[0])
                x="_".join(w for w in y)
                if y[0] not in sp.keys():
                    sp[y[0]]={i:x}                    # chandrayaan-1:chandrayaan_1
                    sp[y[0]][y[0]+" "+'-'+y[1]]=x      #chandrayaan -1:chandrayaan_1
                    sp[y[0]]["".join(w for w in y)]=x  #chandrayaan1:chandrayaan_1
                    sp[y[0]][" ".join(w for w in y)]=x #chandrayaan 1:chandrayaan_1
                    sp[y[0]][y[0]+" "+'-'+" "+y[1]]=x  #chandrayaan - 1:chandrayaan_1
                    sp[y[0]][y[0]+'-'+" "+y[1]]=x      #chandrayaan- 1:chandrayaan_1
                else:
                    sp[y[0]][y[0]+" "+'-'+y[1]]=x
                    sp[y[0]][i]=x
                    sp[y[0]]["".join(w for w in y)]=x
                    sp[y[0]][" ".join(w for w in y)]=x
                    sp[y[0]][y[0]+" "+'-'+" "+y[1]]=x
                    sp[y[0]][y[0]+'-'+" "+y[1]]=x 

            return sp,co

        responses.extend(questions)

        sp,co=co_occurences_with_hyphen(responses)
        
        filename1='series_s.pickle'
        outfile=open(filename1,'wb')
        pickle.dump(series,outfile)
        outfile.close()
        
        with open('phrase_s.csv','w',newline='',encoding='utf-8')as file:
            writer=csv.writer(file)
            writer.writerow(['phrases'])
            for i in set(co):
                writer.writerow([i])

        return sp,filename1
    
    def Phrase_collection(self,filename,entities,sp={},scoring='default',min_count=4,threshold=0.88):
        sc=scoring
        mincount=min_count
        th=threshold
        with open(filename,encoding='utf-8') as myFile:  
            reader = csv.DictReader(myFile)
            responses=[]
            questions=[]
            for row in reader:
                #questlist.append(row['questions'])
                text=row['responses']
                text1=row['questions']
                responses.append(text)
                questions.append(text1)
        
        sentences=[]
        sentences.append([])
        for i in responses:
            [sentences[0].append(w) for w in nltk.word_tokenize(i)]

        pharse_model=Phrases(sentences, min_count=min_count, threshold=threshold,connector_words=ENGLISH_CONNECTOR_WORDS,scoring=scoring)

        phrases=[]
        for phrase, score in pharse_model.find_phrases(sentences).items():
            phrases.append(phrase)
            
        phrase=[]
        for i in phrases:
            y=i.split('_')
            phrase.append(' '.join(w for w in y))
 
        phrases=[]
        names=[]
        for i in entities.keys():
            if len(i[0].split())>1 and i[1] not in ['CARDINAL','DATE','QUANTITY','MONEY']:
                if i[1]=='PERSON':
                    names.append(i[0])
                phrases.append(i[0])
        for i in range(len(phrases)):
            phrases[i]=re.sub(r'the |The ','',phrases[i])
            
        phrase.extend(phrases)
        
        for i in phrase:
            y=i.split()
            if y[0] not in sp.keys():
                sp[y[0]]={i:'_'.join(w for w in y)}
            else:
                sp[y[0]][i]='_'.join(w for w in y)
            
        for i in range(len(questions)):
            questions[i]=self.replace_phrases(sp,questions[i])
            responses[i]=self.replace_phrases(sp,responses[i])
            

        with open('phrase_corpus_s.csv','w',newline='',encoding='utf-8')as file:
            writer=csv.writer(file)
            writer.writerow(['questions','responses'])
            for i in range(len(questions)):
                writer.writerow([questions[i],responses[i]])
                
        with open('phrase_s.csv','a',newline='',encoding='utf-8')as file:
            writer=csv.writer(file)
            writer.writerow(['phrases'])
            for i in phrase:
                writer.writerow([i])

        filename1='phrases_s.pickle'
        outfile=open(filename1,'wb')
        pickle.dump(sp,outfile)
        outfile.close()

        return 'phrase_corpus_s.csv','phrases_s.pickle','phrase_s.csv'
    
    def replace_phrases(self,sp,sentences):
        for i in sentences.split():
            if '-' in i:
                y=i.split('-')
                for k in sp.keys():
                    if k.lower()==y[0] or k==y[0]:
                        d=sp[k]
                        for m in d.keys():
                            if m.lower() in i:
                                sentences=sentences.replace(m.lower(),d[m])
                            elif m in i:
                                sentences=sentences.replace(m,d[m])
            else:
                for k in sp.keys():
                    if k.lower()==i or k==i:
                        d=sp[k]
                        for m in d.keys():
                            if m.lower() in sentences:
                                sentences=sentences.replace(m.lower(),d[m])
                            elif m in sentences:
                                sentences=sentences.replace(m,d[m])
        return sentences


# In[5]:


def phrases_replace(sentences):
    '''
    for i in sentences.split():
        if '-' in i:
            y=i.split('-')
            for k in sp.keys():
                if k.lower()==y[0] or k==y[0]:
                    d=sp[k]
                    for m in d.keys():
                        if m.lower() in i:
                            sentences=sentences.replace(m.lower(),d[m])
                        elif m in i:
                            sentences=sentences.replace(m,d[m])
        else:
            for k in sp.keys():
                if k.lower()==i or k==i:
                    d=sp[k]
                    for m in d.keys():
                        if m.lower() in sentences:
                            sentences=sentences.replace(m.lower(),d[m])
                        elif m in sentences:
                            sentences=sentences.replace(m,d[m])
    '''
    return sentences


# ### Synonyms generation

# In[6]:


def extract_corpus_file(corpus_file):
    dataset=pd.read_csv(corpus_file) #raw corpus
    questions=[] #list of questions
    responses=[] #list of responses
    questions.extend(list(dataset['questions']))
    responses.extend(list(dataset['responses']))
    return questions,responses

def remove_punctuation(sentence): #function to remove any punctuations from the sentence
    sentence=re.sub(r'\'s\?','',sentence)
    sentence=re.sub(r'[^\w\s]',' ',sentence)
    sentence=sentence.lower()
    return sentence

def remove_stopwords(sentence): #function to tokenize such that there are no stopwords
    stop_words=set(stopwords.words('english'))
    return [word for word in nltk.word_tokenize(sentence) if word not in stop_words]

def lemmatize_words(tokens): #function to return the lemmas with
    tokens=[lemmatizer.lemmatize(token) for token in tokens]
    return [lemmatizer.lemmatize(token,pos='v') for token in tokens]

def find_distinct(documents):
    distinct=[]
    for doc in documents: #find all the distinct words from the tokens
        for word in doc:
            if word not in distinct:
                if not (word.isalpha() and len(word)==1):
                    distinct.append(word)
    distinct.sort() #sort the terms in ascending order
    return distinct

def document_cleaning(documents):
    for docidx in range(len(documents)): #cleaning questions
        documents[docidx]=lemmatize_words(remove_stopwords(remove_punctuation(documents[docidx])))
        for termidx in range(len(documents[docidx])):
            if '_' in documents[docidx][termidx]:
                documents[docidx][termidx]=re.sub(r'\_',' ',documents[docidx][termidx]) 
    return documents

def find_corpus_distinct_words(que_distinct_words,res_distinct_words):
    distinct_words=[]
    distinct_words.extend(que_distinct_words)
    distinct_words.extend(res_distinct_words)
    distinct_words=list(set(distinct_words))
    distinct_words.sort()
    return distinct_words

def generate_distinct_words_file(distinct_words):
    obj=pd.DataFrame(distinct_words)
    obj.to_csv('distinct_words_s.csv',index=False,header=['words'])

def generate_distinct_words(corpus_file):
    questions,responses=extract_corpus_file(corpus_file)
    que_documents=document_cleaning(questions)
    res_documents=document_cleaning(responses)
    que_distinct_words=find_distinct(que_documents)
    res_distinct_words=find_distinct(res_documents)
    distinct_words=find_corpus_distinct_words(que_distinct_words,res_distinct_words)
    generate_distinct_words_file(distinct_words)

def create_pickle_file(name,data):
    with open(name, 'wb') as handle:
        pickle.dump(data,handle, protocol=2)

#seperating words and numbers 
def seperation_words_nums(words_list):
    num=[]
    words=[]
    for i in words_list:
        if i.isdigit():
            num.append(i)
        elif len(i)>1 :
            words.append(i)
    return num,words       

#generating synonyms for the numbers
def num_synonyms_generation(dt):
    nd={}
    for j in dt:
        num=[]
        if(j.isdigit()):
            num.append(num2words(j))
            nd[j]=num
    return nd

#generating synonyms for the words
def synonyms_generation(tokens):
    synset={}
    dp=[]
    for wrd in tokens: #getting the words
            synl=[]
            for syn in wordnet.synsets(wrd):  #getting the synonyms set for the word
                    if syn:
                        for l in syn.lemmas():
                            sq=l.name().lower()
                            sq=sq.replace("_"," ")
                            sq=sq.replace(",","")
                            sq=sq.replace("-"," ")
                            if (sq not in tokens) and (sq!=wrd) and (sq not in dp) :
                                dp.append(sq)
                                synl.append(sq)  
            if len(synl)!=0:   #if synonyms list(synl) is not empty
                        l1=[]
                        synl.append(wrd)  #adding the word to the synonyms list(synl)
                        for i in synl:    #removing redundant words in synonyms list(synl)
                            if i not in l1:    
                                l1.append(i)
                        synset[wrd]=l1    #creating a dictionary key value pair using synset dictionary
            else:
                    synl.append(wrd)      #if the synonyms list(synl) is empty
                    synset[wrd]=synl      # adding the word itself to the synonyms list and generating key value pair            
    return synset

#cleaning the dictionary by removing unnecessary symbols
def cleaning_dictionary(dx):
    ky=dx.keys()
    for k in ky:
        l1=[]
        l2=[]
        l1=dx[k]
        for i in l1:
            i=i.replace("_"," ")
            i=i.replace(",","")
            i=i.replace("-"," ")
            l2.append(i)
        dx[k]=l2
    return dx

def make_list(dt):
    rk=dt.keys()
    lx=[]
    for r in rk:
        j=dt[r]
        if j not in lx: 
            lx.append(j)
    return lx

def extract_distinct_file(file):
    dataset=pd.read_csv(file)
    words=[] 
    words.extend(list(dataset['words']))
    return words

def generate_synonym_file(corpus_dictionary):
    create_pickle_file('word_synonymlist_s.pickle',corpus_dictionary) #distinct_word and its synonyms list

def generate_synonyms(distinct_words_file):
    words=extract_distinct_file(distinct_words_file)
    num_list,words_list=seperation_words_nums(words) #seperate numbers and words
    num_synonyms=num_synonyms_generation(num_list) #number synonyms
    words_synonyms=synonyms_generation(words_list) #words synonyms
    corpus_dictionary={**words_synonyms,**num_synonyms} #merging two dictionaries for forming corpus dictionaries
    corpus_dictionary=cleaning_dictionary(corpus_dictionary)
    generate_synonym_file(corpus_dictionary)


# ### One word description

# In[7]:


def get_nouns_list(intents):
    lt=['NNP']
    nouns_list=[]
    for i in intents:
        for j in i:
            for k in j:
                if k[1] in lt and k[0].lower() not in stopwords.words('english'):
                    if k[0] not in nouns_list:
                        nouns_list.append(k[0])
    return nouns_list


# In[8]:


def get_definations_s(i,definations_s):# i is list of nouns in caps or mixed
        
    def get_from_web(c):
        try:
            results=wikipedia.search(c)
            info=wikipedia.page(results[0]).summary
        except:
            info='not available'
            pass
        return info
    for j in i:
        text=get_from_web(j)
        if text!='not available':
            text=text.replace('\n','')
            text=text.encode('ascii',errors='ignore').decode('ascii')
            sent_tokens=[]
            [sent_tokens.append(sent) for sent in nltk.sent_tokenize(text)]
            for k in range(2):
                try:
                    if j.lower() not in definations_s.keys():
                        definations_s[j.lower()]=sent_tokens[k]+' '
                    else:
                        definations_s[j.lower()]+=sent_tokens[k]+' '
                except:
                    pass
    filename='one_word_descriptions_s.pickle'
    output=open(filename,'wb')
    pickle.dump(definations_s,output)
    output.close()   
    return definations_s

def definations_s(filename,series_file,noun_list_file): # filename csv file of corpus,series is pickle file of series list,noun_list is pickle file
    
    def remove_stopwords(sentence): #STOPWORDS REMOVAL
        stop_words=set(stopwords.words('english'))
        return [word for word in nltk.word_tokenize(sentence) if word not in stop_words]
    def remove_punctuation(sentence): #PUNCTUATION REMOVAL
        sentence=re.sub(r'\'s\?','',sentence)
        sentence=re.sub(r'[^\w\s\-]',' ',sentence)
        return sentence
    definations_s={}
    for i in range(len(questionlist)):
        intent=remove_stopwords(remove_punctuation(questionlist[i].lower()))
        if len(intent)==1:
            if intent[0] not in definations_s.keys():
                definations_s[intent[0]]=responselist[i]
    
    nouns = set()
    for i in definations_s.keys():
        nouns.add(i)
    nouns = set()
    with open('series_s.pickle', 'rb') as file: 
        series = pickle.load(file)
        file.close()
    for i in series:
        nouns.add(i)
    for i in noun_list_file:
        nouns.add(i)
        
    return get_definations_s(nouns,definations_s)


# In[9]:


#one word description
def one_word_description(int_set):
    for word in int_set:
        if word in sin_word_def:
            return sin_word_def[word]
    return


# ### Entity extraction with labels

# In[10]:


def create_pickle(file, data):
    open_file = open(file, 'wb')
    pickle.dump(data, open_file)
    open_file.close()

def generate_entities_and_labels_dictionary(corpus_file):
    #load spacy model
    nlp = spacy.load("en_core_web_sm")

    arr = []     #question_entities
    arr1 = []    #response_entities
    i = 0

    for i in range(len(raw_questions)):
        if(raw_questions[i][-1].isalnum() == False):
            doc = nlp(raw_questions[i][:-1])
        else:
            doc = nlp(raw_questions[i])
        arr.append({})
        for ent in doc.ents:
            arr[i][str(ent)] = ent.label_
        
        if(raw_responses[i][-1].isalnum() == False):
            doc = nlp(raw_responses[i][:-1])
        else:
            doc = nlp(raw_responses[i])
        arr.append({})
        for ent in doc.ents:
            arr[i][str(ent)] = ent.label_
        i += 1

    dis_que = {}
    dis_res = {}

    for i in range(0, len(arr)):
        dic = arr[i]
        for j in dic:                
            k = []
            k.append(j)
            k.append(dic[j])
            k = tuple(k)
            if k not in dis_que:
                dis_que[k] = {i}
            else:
                dis_que[k].add(i)

    for i in range(0, len(arr1)):
        dic = arr1[i]
        for j in dic:
            k = []
            k.append(j)
            k.append(dic[j])
            k = tuple(k)
            if k not in dis_res:
                dis_res[k] = {i}
            else:
                dis_res[k].add(i)

    create_pickle('entities_and_labels_questions_dictionary_s.pickle', dis_que)
    create_pickle('entities_and_labels_responses_dictionary_s.pickle', dis_res)


# ### Entities mapping to corpus indices

# In[11]:


def generate_entities_mapped_to_corpus_dictionary(corpus_file):
    #load spacy model
    nlp = spacy.load("en_core_web_sm")

    ent_set = set()

    for i in range(len(questionlist)):
        if(questionlist[i][-1].isalnum() == False):
            doc = nlp(questionlist[i][:-1])
        else:
            doc = nlp(questionlist[i])
        for ent in doc.ents:
            ent_set.add(str(ent).lower())

        if(responselist[i][-1].isalnum() == False):
            doc = nlp(responselist[i][:-1])
        else:
            doc = nlp(responselist[i])
        for ent in doc.ents:
            ent_set.add(str(ent).lower())
    
    dis_ent_dic_que = {}
    dis_ent_dic_res = {}

    for i in range(0, len(questionlist)):
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        words_list = tokenizer.tokenize(questionlist[i])
        
        for word in words_list:
            j = word.lower()
            if j in ent_set:
                if j not in dis_ent_dic_que:
                    dis_ent_dic_que[j] = {i}
                else:
                    dis_ent_dic_que[j].add(i)
        
        words_list = tokenizer.tokenize(responselist[i])
        for word in words_list:
            j = word.lower()
            if j in ent_set:
                if j not in dis_ent_dic_res:
                    dis_ent_dic_res[j] = {i}
                else:
                    dis_ent_dic_res[j].add(i)   
                
    create_pickle('entities_mapped_to_questions_dictionary_s.pickle', dis_ent_dic_que)
    create_pickle('entities_mapped_to_responses_dictionary_s.pickle', dis_ent_dic_res)


# ### Entity extraction

# In[12]:


#entity extraction
def entity_extraction(user_input):
    shared_items_dic = {}
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    words_list = tokenizer.tokenize(user_input)
    
    for word1 in words_list:
        word = word1.lower()
        if word in arr:
            if word not in shared_items_dic:
                shared_items_dic[word] = set()
            shared_items_dic[word] = shared_items_dic[word].union(arr[word])

        if word in arr1:
            if word not in shared_items_dic:
                shared_items_dic[word] = set()
            shared_items_dic[word] = shared_items_dic[word].union(arr1[word])
        
    if(len(shared_items_dic) == 0):
        return -1      
        #bot_response
    else:
        return shared_items_dic


# ### Entiy extraction using spacy model for the user query

# In[13]:


#entity extraction
def entity_extraction1(user_input):
    dic1 = get_entities(user_input)

    if(len(dic1) == 0):
        return -1      
        #bot_response
    else:
        shared_items_dic = compare_entities(dic1)
        if(len(shared_items_dic) == 0):
            return -2    
            #unknown_entity
        else:
            return shared_items_dic

def get_entities(user_input):
    nlp = spacy.load("en_core_web_sm")
    dic1 = {}
    gen_ents = nlp(user_input)
    for ent in gen_ents.ents:
        dic1[str(ent)] = ent.label_
    print("Entities: ", dic1)
    return dic1

def compare_entities(dic1):
    shared_items_dic = {}
    
    for i in dic1:
        j = i.lower()
        
        if j in arr:
            if j not in shared_items_dic:
                shared_items_dic[j] = set()
            for k in arr[j]:
                shared_items_dic[j].add(k)
        
        if j in arr1:
            if j not in shared_items_dic:
                shared_items_dic[j] = set()
            for k in arr1[j]:
                shared_items_dic[j].add(k)

    return shared_items_dic


# ### Corpus dictionary

# In[14]:


def build_corpus_dictionary(corpus_file):
    if os.path.exists("corpus_s.txt"):
        os.remove("corpus_s.txt")
    
    #text file that stores the distinct words
    corpus_dict = open("corpus_s.txt", "a", encoding = 'cp1252')
    words_set = set()

    #this reads the corpus csv file
    with open(corpus_file, 'r', encoding = "utf8") as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            for col in row:
                tokenizer = nltk.RegexpTokenizer(r"\w+")
                words_list = tokenizer.tokenize(col)
                for word in words_list:
                    if word.isnumeric() == False:
                        words_set.add(word)
                        words_set.add(word.lower())

    #this reads the generated synonyms
    syn = pd.read_pickle('word_synonymlist_s.pickle')
    for items in syn:
        if items.isnumeric() == False: 
            words_set.add(items)
            for word in syn[items]:
                if word.isnumeric() == False: 
                    #if the word contains space(' ')
                    j = 0
                    for i in range(0, len(word)):
                        if word[i] == ' ':
                            if word[j:i].isnumeric() == False:
                                words_set.add(word[j:i])
                                j = i + 1
                    words_set.add(word[j:])

    #this reads the detected abbreviaions
    abb = pd.read_pickle('abbreviations_s.pickle')
    for item in abb:
        for word in abb[item]:
            if word.isnumeric() == False:
                words_set.add(word)
                words_set.add(word.lower())
                #if the word contains hyphen(-)
                for i in range(0, len(word)):
                    if word[i] == '-':
                        if word[0:i].isnumeric() == False:
                            words_set.add(word[0:i])
                            words_set.add(word[0:i].lower())
                        if word[i+1:].isnumeric() == False:    
                            words_set.add(word[i+1:])
                            words_set.add(word[i+1:].lower())
            if abb[item][word].isnumeric() == False:
                words_set.add(abb[item][word])
                words_set.add(abb[item][word].lower())

    #this reads the detected phrases
    di = pd.read_pickle('phrases_s.pickle')
    for i in di:
        if i.isnumeric() == False:
            words_set.add(i)
            words_set.add(i.lower())

        for word in di[i]:
            if word.isnumeric() == False:  
                #if the word contains hyphen
                if '-' in word:
                    words_set.add(word)
                    words_set.add(word.lower())   
                #if the word contains space, underscore, hyphen
                j = 0
                for i in range(0, len(word)):
                    if word[i] == ' ' or word[i] == '_' or word[i] == '-':
                        if word[j:i].isnumeric() == False:
                            words_set.add(word[j:i])
                            words_set.add(word[j:i].lower())
                            j = i + 1
                words_set.add(word[j:])
                words_set.add(word[j:].lower())

    #this reads the one word descriptions
    with open('one_word_descriptions_s.pickle', 'rb') as read_obj:
        pkl_reader = pickle.load(read_obj)
        for items in pkl_reader.items():
            tokenizer = nltk.RegexpTokenizer(r"\w+")
            words_list = tokenizer.tokenize(str(items))
            for word in words_list:
                if word.isnumeric() == False:
                    words_set.add(word)
                    words_set.add(word.lower())

    create_pickle("words_set_s.pickle", list(words_set))
    
    for word in words_set:
        try:
            word = word.encode("cp1252").decode("cp1252")
            corpus_dict.write(word)
            corpus_dict.write("\n")
        except:
            pass

    corpus_dict.write('fullform')
    corpus_dict.write("\n")

    corpus_dict.close()


# In[15]:


# def build_corpus_dictionary_mac(corpus_file):
#     if os.path.exists("corpus_s.txt"):
#         os.remove("corpus_s.txt")
    
#     #text file that stores the distinct words
#     corpus_dict = open("corpus_s.txt", "a", encoding = 'utf8')
#     words_set = set()

#     #this reads the corpus csv file
#     with open(corpus_file, 'r', encoding = "utf8") as read_obj:
#         csv_reader = reader(read_obj)
#         for row in csv_reader:
#             for col in row:
#                 tokenizer = nltk.RegexpTokenizer(r"\w+")
#                 words_list = tokenizer.tokenize(col)
#                 for word in words_list:
#                     if word.isnumeric() == False:
#                         words_set.add(word)
#                         words_set.add(word.lower())

#     #this reads the generated synonyms
#     syn = pd.read_pickle('word_synonymlist_s.pickle')
#     for items in syn:
#         if items.isnumeric() == False: 
#             words_set.add(items)
#             for word in syn[items]:
#                 if word.isnumeric() == False: 
#                     #if the word contains space(' ')
#                     j = 0
#                     for i in range(0, len(word)):
#                         if word[i] == ' ':
#                             if word[j:i].isnumeric() == False:
#                                 words_set.add(word[j:i])
#                                 j = i + 1
#                     words_set.add(word[j:])

#     #this reads the detected abbreviaions
#     abb = pd.read_pickle('abbreviations_s.pickle')
#     for item in abb:
#         for word in abb[item]:
#             if word.isnumeric() == False:
#                 words_set.add(word)
#                 words_set.add(word.lower())
#                 #if the word contains hyphen(-)
#                 for i in range(0, len(word)):
#                     if word[i] == '-':
#                         if word[0:i].isnumeric() == False:
#                             words_set.add(word[0:i])
#                             words_set.add(word[0:i].lower())
#                         if word[i+1:].isnumeric() == False:    
#                             words_set.add(word[i+1:])
#                             words_set.add(word[i+1:].lower())
#             if abb[item][word].isnumeric() == False:
#                 words_set.add(abb[item][word])
#                 words_set.add(abb[item][word].lower())

#     #this reads the detected phrases
#     di = pd.read_pickle('phrases_s.pickle')
#     for i in di:
#         if i.isnumeric() == False:
#             words_set.add(i)
#             words_set.add(i.lower())

#         for word in di[i]:
#             if word.isnumeric() == False:  
#                 #if the word contains hyphen
#                 if '-' in word:
#                     words_set.add(word)
#                     words_set.add(word.lower())   
#                 #if the word contains space, underscore, hyphen
#                 j = 0
#                 for i in range(0, len(word)):
#                     if word[i] == ' ' or word[i] == '_' or word[i] == '-':
#                         if word[j:i].isnumeric() == False:
#                             words_set.add(word[j:i])
#                             words_set.add(word[j:i].lower())
#                             j = i + 1
#                 words_set.add(word[j:])
#                 words_set.add(word[j:].lower())

#     #this reads the one word descriptions
#     with open('one_word_descriptions_s.pickle', 'rb') as read_obj:
#         pkl_reader = pickle.load(read_obj)
#         for items in pkl_reader.items():
#             tokenizer = nltk.RegexpTokenizer(r"\w+")
#             words_list = tokenizer.tokenize(str(items))
#             for word in words_list:
#                 if word.isnumeric() == False:
#                     words_set.add(word)
#                     words_set.add(word.lower())

#     create_pickle("words_set_s.pickle", list(words_set))

#     corpus_dict.write('fullform')
#     corpus_dict.write("\n")

#     corpus_dict.close()


# # ### Spell checker

# # In[16]:


#checks the spellings of user input and corrects them
def spell_check(user_input): 
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    words_list = tokenizer.tokenize(user_input)
    
    for word in words_list:
        if word not in corpus_dic:
            if word.isnumeric() == False:
                if standard_dict.check(word) == False:
                    suggestions = corpus_dict.suggest(word)
                    if(len(suggestions) != 0):
                        user_input = user_input.replace(word, suggestions[0])
                    else:
                        user_input = user_input.replace(word, standard_dict.suggest(word)[0])

    return user_input


# ### Full form check

# In[17]:


#returns the full form of the acronym
def find_fullform(user_input):
    for i in abbreviations_dict:
        j = " " + abbreviations_dict[i]
        if j in user_input or j.lower() in user_input:
            return i


# ### Intents and synonyms

# In[18]:


# A method to get Hypernyms of a given word
def getHyperNyms(word):
    word_hypernyms=set()
    syns=wordnet.synsets(word)
    for i,j in enumerate(syns):
        [word_hypernyms.add(elt) for elt in (list(chain(*[l.lemma_names() for l in j.hypernyms()])))]
    return word_hypernyms


# In[19]:


# A method to find path similarity between two English words based on their root tree of synonyms. 
# If words have common synonyms return 1, else find the max path similarity between pairs of synonyms drawn-one from each word.
def getPathSimilarity(word1, word2):
    syn1 = wordnet.synsets(word1)
    syn2 = wordnet.synsets(word2)
    commonsyns=set(syn1).intersection(set(syn2))
    if commonsyns:
        return 1
    # A word may have multiple synsets, so need to compare each synset of word1 with each synset of word2
    scores=[s1.path_similarity(s2) for s2 in syn2 for s1 in syn1 if s1.path_similarity(s2)]
    if scores:
        return max(scores) 
    else:
        return 0


# In[20]:


# A method to load questions and responses to a list so that one can see the best match question/response without altered to user response
def getQuestionseAndResponseList(qnfile, questions_col_name, response_col_name):
    with open(qnfile,encoding="utf8") as myFile:  
        reader = csv.DictReader(myFile)
        questionlist = []
        responselist=[]
        for row in reader:
            qn_text=row[questions_col_name]
            res_text=row[response_col_name]
            questionlist.append(qn_text)
            responselist.append(res_text)
        return questionlist,responselist


# In[21]:


# An intent of a question/response is the set of alpha numeric non stop words of that question/response along with their pos tags. Replace '-' with space so that we can consider them as two words. The method returns intents of question/response and stopword set for each question/response as a pair.
def getIntents(raw_list):
    intents=[]
    for elt in raw_list:
        sents=sent_tokenize(elt)
        intent_tags=set()
        stop_words=set()
        for sent in sents:
            sent=sent.replace('-',' ')
            tags=pos_tag(word_tokenize(sent))
            for tag in tags:
                if tag[0].isalnum():
                    if tag[0].lower() not in stopwords.words():
                        intent_tags.add(tag)
                    else:
                        stop_words.add(tag)
        intents.append((intent_tags,stop_words))
    return intents


# In[22]:


# A method to get Synonyms and Hypernyms of a given word with POS tag
def getSynonymSet(word, pos_tag):
    if pos_tag=='j':
        pos_tag='a'
    if pos_tag not in ['n','a','r','v','s']:
        return
    syn = set()
    for synset in wordnet.synsets(word,pos_tag):
        for lemma in synset.lemmas():
            syn.add(lemma.name())
        hypos=synset.hypernyms()
        for hypo in hypos:
            for lemma in hypo.lemmas():
                syn.add(lemma.name())
    return syn


# In[23]:


# A method to generate dictionary of unique words and their sysnsets including hypernyms
def getSynonymDictionary(intents, syn_dict):
    for intent in intents:
    # Each intent has two sets one for non-stopwords and one for stopwords. 
        for word in intent[0]:
            token=word[0].lower()
            pos_tag=word[1][0].lower()
            if not (token,pos_tag) in syn_dict:
                syns=getSynonymSet(token, pos_tag)
                if not syns:
                    syns=set()
                    syns.add(token)
                syn_dict[(token, pos_tag)]=syns
    return syn_dict


# In[24]:


# The method creates a dictionary with word as key and set of tuples where each tuple consists index of a question/response, the word in question/response for which key is synonym, and POS tag of the word in question/response. All words and their synonyms of questions_intents/response_intents and questions_stopwords/response_intents.

def getWordTOIntentsDict(intents,syn_dict):
    word_to_intents_dict={}
    for ind in range(0,len(intents)):
        # Each intent has two sets one for non-stopwords and one for stopwords. 
        for word in intents[ind][0]:
            token=word[0].lower()
            pos_tag=word[1][0].lower()
            if not (token,pos_tag) in syn_dict:
                syns=getSynonymSet(token, pos_tag)
                if not syns:
                    syns=set()
                    syns.add(token)
                syn_dict[(token, pos_tag)]=syns
            syns=syn_dict[(token, pos_tag)]
            for syn in syns:
                if syn == token:
                    ps = 1
                else:
                    ps = getPathSimilarity(syn, token)
                if (syn,pos_tag) in word_to_intents_dict:
                    word_to_intents_dict[(syn,pos_tag)].add((ind,token,word[1],ps))
                else:
                    word_to_intents_dict[(syn,pos_tag)]={(ind,token,word[1],ps)}
    return word_to_intents_dict


# In[25]:


# The method process user response and returns user intent and the user stopwords.
def getUserIntentAndStopWords(user_input):
    sents=sent_tokenize(user_input)
    user_intent=set()
    user_stopwords=set()
    for sent in sents:
        sent=sent.replace('-',' ')
        tags=pos_tag(word_tokenize(sent))
        for word_and_pos in tags:
            if word_and_pos[0].isalnum():
                if word_and_pos[0].lower() not in stopwords.words():
                    user_intent.add(word_and_pos)
                else:
                    user_stopwords.add(word_and_pos)
    return user_intent,user_stopwords


# In[26]:


# The method retrieve all questions/responses in which user intent words are prsesent. It builds a dictionary for which key is word in user intent and value is again a dictionary for which key is question/response index in which the word (key) or its synonym present and value is the path similarity between the matching words in user intent and question/response intent.To match words How many letters to match for POS tag should be given as input 
def getIntentsInetrsectingUserIntent(user_intent,WordTOIntentDict, POS_Tag_len_to_match,syn_dict,ent):
    intent_set={}
    for tag in user_intent:
        token=tag[0].lower()
        pos_tag=tag[1][0].lower()
        if not (token,pos_tag) in syn_dict:
            syns=getSynonymSet(token, pos_tag)
            if not syns:
                syns=set()
                syns.add(token)
            syn_dict[(token, pos_tag)]=syns
        syns=syn_dict[(token, pos_tag)]
        for syn in syns:
            if (syn,pos_tag) in WordTOIntentDict:
                for qn in WordTOIntentDict[(syn,pos_tag)]:
                    if tag[1][0:POS_Tag_len_to_match]==qn[2][0:POS_Tag_len_to_match]:
                        ps=qn[3]
                        if token not in intent_set:
                            intent_set[token]={}
                            intent_set[token][qn[0]]=ps
                        elif qn[0] not in intent_set[token]:
                            intent_set[token][qn[0]]=ps
                        elif intent_set[token][qn[0]]<ps:
                            intent_set[token][qn[0]]=ps
        
    if(type(ent) == type({})):
        #one word descriptions
        if(len(ent) == 1 and len(intent_set) == 1):
            if ent.keys() == intent_set.keys():
                word = one_word_description(set(ent.keys()))
                if not word:
                    return
                else:
                    return word
        
        #entities intersection   
        z = -1
        for i in ent:
            if z == -1:
                z = 1
                ent_int = ent[i]
            else:
                ent_int = ent_int.intersection(ent[i])
        
        #intents intersection entities
        int_set = intents_intersection_entities(intent_set, ent_int)
        
        return int_set
    
    return intent_set


# In[27]:


def intents_intersection_entities(intent_set, ent_int): 
    intent_indices = set()
    for i in intent_set:
        for j in intent_set[i]:
            intent_indices.add(j)

    s = intent_indices.intersection(ent_int)

    for i in intent_set:
        for j in list(intent_set[i]):
            if j not in s:
                del intent_set[i][j]
        
    return intent_set


# In[28]:


# The method takes questions/responses intersecting with user intent and returns all those questions/responses which share the intent with user intent more than the chosen threshold. It returns a dictionary with key as question/response index and value as match score of that question/response with user intent. (Here the score is computed based on cumulative path similarity based on their shared intent words) 
def getMatchingIntentsForUserIntent(intent_set,user_intent, match_threshold,min_lim=2,max_lim=5):
    if not intent_set:
        return
    if len(user_intent)<=min_lim:
        min_match=len(user_intent)
    else:
        min_match=len(user_intent)*match_threshold
    threshold=min_match
    if threshold>max_lim:
        threshold=max_lim
    intent_score={}
    for item in intent_set:
        for qn in intent_set[item]:
            if qn in intent_score:
                intent_score[qn]=intent_score[qn]+intent_set[item][qn]
            else:
                intent_score[qn]=intent_set[item][qn]
    intents_to_be_deleted=[item for item in intent_score if round(intent_score[item])<threshold]
    for item in intents_to_be_deleted:
        del intent_score[item]
    return intent_score


# In[29]:


# The method enhanses the intent scores based on matching of stop words in questions/responses and user responses   
def enhanseMatchingIntentScoresWithUserStopwords(user_stopwords,intents, intent_score):
    for tag in user_stopwords:
        user_stop_word=tag[0].lower()
        if user_stop_word.isalnum():
            for item in intent_score:
                for intent_stop_word in intents[item][1]:
                    if intent_stop_word[1]==tag[1]:
                        if intent_stop_word[0]==user_stop_word:
                            intent_score[item]=intent_score[item]+1
                            break
                        elif getPathSimilarity(user_stop_word, intent_stop_word[0])==1:
                            intent_score[item]=intent_score[item]+1
                            break
    return intent_score


# In[30]:


# The method takes a list of matched responses and returns all matches with max matching score
def getmaxMatchingResponses(user_stopwords,intent_score,intents, user_response_len):
    if not intent_score:
        return
    else:
        intent_score=enhanseMatchingIntentScoresWithUserStopwords(user_stopwords,intents, intent_score)
        bot_response_list=[]
        matching_qns=[(qn,intent_score[qn]) for qn in intent_score]
        matching_qns.sort(key=lambda x:x[1], reverse=True)
        bot_response_list.append((matching_qns[0],matching_qns[0][1]/(len(intents[matching_qns[0][0]][0])+len(intents[matching_qns[0][0]][1])+user_response_len-matching_qns[0][1])))
        if len(matching_qns)>1:
            for i in range(1,len(matching_qns)):
                if matching_qns[i][1]==matching_qns[i-1][1]:
                    bot_response_list.append((matching_qns[i],matching_qns[i][1]/(len(intents[matching_qns[i][0]][0])+len(intents[matching_qns[i][0]][1])+user_response_len-matching_qns[i][1])))
                else:
                    break
    bot_response_list.sort(key=lambda x:x[1],reverse=True)
    bot_response_list=[ res for res in bot_response_list if res[1]==bot_response_list[0][1]]
    return bot_response_list


# In[31]:


# The method finds the chatbot response for user response
def getChatBotResponse(user_response,intents,POS_Tag_len_to_match,word_to_intents_dict,match_threshold,min_lim,max_lim,syn_dict,ent):
    user_intent,user_stopwords=getUserIntentAndStopWords(user_response)
    user_response_len=len(user_intent)+len(user_stopwords)

    intent_set=getIntentsInetrsectingUserIntent(user_intent,word_to_intents_dict, POS_Tag_len_to_match,syn_dict,ent)
    if not intent_set:
        return
    elif type(intent_set) == type(''):
        return intent_set
    intent_score=getMatchingIntentsForUserIntent(intent_set,user_intent, match_threshold,min_lim,max_lim)
    if not intent_score:
        return
    perfect_Matched_intents={}
    for item in intent_score:
        if intent_score[item]==len(user_intent):
            perfect_Matched_intents[item]=intent_score[item]
    if perfect_Matched_intents:
        Bot_response=getmaxMatchingResponses(user_stopwords,perfect_Matched_intents,intents, user_response_len)
        return True,Bot_response
    else:
        Bot_response=getmaxMatchingResponses(user_stopwords,intent_score,intents, user_response_len)
        return False,Bot_response


# In[32]:


# The method finds closest match from question and response intents to user intent
def getClosestBotResponse(user_response,qn_intents,res_intents,POS_Tag_len_to_match,word_to_qn_intents_dict,word_to_res_intents_dict,match_threshold,min_lim,max_lim,syn_dict,ent):
    qn_matches=getChatBotResponse(user_response,qn_intents,POS_Tag_len_to_match,word_to_qn_intents_dict,match_threshold,min_lim,max_lim,syn_dict,ent)
    if type(qn_matches) == type(''):
        return qn_matches
    elif qn_matches:
    # If perfect from question intents is found return all perfect match question indices
        if qn_matches[0]:
            return qn_matches[1]
        else:
            res_matches=getChatBotResponse(user_response,res_intents,POS_Tag_len_to_match,word_to_res_intents_dict,match_threshold,min_lim,max_lim,syn_dict,ent)
            # If perfect from response intents is found return all perfect match response indices
            if type(res_matches) == type(''):
                return res_matches
            elif res_matches:
                if res_matches[0]:
                    return res_matches[1]
                elif qn_matches[1][0][0][1]>=res_matches[1][0][0][1]:
                    return qn_matches[1]
                else:
                    return res_matches[1]
            else:
                return qn_matches[1]
    else:
        res_matches=getChatBotResponse(user_response,res_intents,POS_Tag_len_to_match,word_to_res_intents_dict,match_threshold,min_lim,max_lim,syn_dict,ent)
        # If perfect from response intents is found return all perfect match response indices
        if type(res_matches) == type(''):
            return res_matches
        elif res_matches:
            return res_matches[1]
        else:
            return


# In[33]:


# The method saves responses to greet
def greeting(sentence):
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    words_list = tokenizer.tokenize(sentence)
    
    GREETING_INPUTS = ("hello", "hi", "greetings", "what's up","hey",)
    GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
    greet_inputs = ['how are you', 'how do you do', 'how are you doing']
    greet_responses = ['Good', 'Very well thanks', 'Fine and you', 'I am doing well', 'I am doing well How are you']
    for word in words_list:
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
    
    tokenizer = nltk.RegexpTokenizer(r"[\w+' ']+")
    words_list = tokenizer.tokenize(sentence)
    
    if words_list[0].lower() in greet_inputs:
        return random.choice(greet_responses)


# ### Test chatbot

# In[34]:


# The method tests the chatbot for given corpus
def TestChatbot(raw_questions,raw_responses,qn_intents,res_intents,POS_Tag_len_to_match,word_to_qn_intents_dict,word_to_res_intents_dict,questionlist,responselist,match_threshold,min_lim,max_lim,syn_dict):
    print("ROBO: My name is ISRO Robo. I will answer your general queries about ISRO. Let us chat. If you want to exit, type Bye!")
    ar = []
    a = 0
    count = 0
    start_time = datetime.now()
    while(a < len(raw_questions)):
        print("-------------------------------------------------------------------------------------------------")
        user_response = raw_questions[a]
        print("USER: ", user_response)

        #spell check
        user_response = spell_check(user_response)
        #print("SPELL: " + user_response)

        if(user_response.lower() == 'bye' or user_response.lower() == 'no'):
            print("ROBO: Bye! take care...")
            break
        elif (user_response == 'thanks' or user_response == 'thank you' ):
            print("ROBO: You are welcome... Do you have any other question?")
        else:
            grt = greeting(user_response)
            #greeting
            if(grt != None):
                print("ROBO: ", grt)
                count += 1
            elif 'fullform' in user_response or 'full form' in user_response:
                #full form check
                response = find_fullform(user_response)
                if not response:
                    print("ROBO: " + "Sorry I don't have enough information to answer you")
                    ar.append(a)
                else:
                    print("ROBO: " + response)
                    count += 1
            else:
                #abbreviation replace
                user_response = abbreviation_check(user_response)
                #print("ABB: " + user_response)

                #entity extraction
                ent = entity_extraction(user_response)

                #find for matching entities
                #bot_response
                bot_response_list = getClosestBotResponse(user_response,qn_intents,res_intents,POS_Tag_len_to_match,word_to_qn_intents_dict,word_to_res_intents_dict,match_threshold,min_lim,max_lim,syn_dict,ent)
                if not bot_response_list:
                    print("ROBO: " + "Sorry I don't have enough information to answer you")
                    ar.append(a)
                elif type(bot_response_list) == type(''):
                    #one word description
                    print("ROBO: " + bot_response_list)
                    count += 1
                elif len(bot_response_list) == 1:
                    print("ROBO: " + raw_responses[bot_response_list[0][0][0]])
                    if raw_responses[bot_response_list[0][0][0]] in res[ques.index(raw_questions[a])]:
                        count += 1
                    else:
                        #different bot response
                        ar.append(a)
                elif len(bot_response_list) > 1:
                    print("ROBO: " + "I am afraid that my response may not be precise. I will provide you relevant information.") 
                    q = -1
                    for k in range(0,len(bot_response_list)):
                        print(raw_responses[bot_response_list[k][0][0]])
                        if raw_responses[bot_response_list[k][0][0]] in res[ques.index(raw_questions[a])]:
                            if q == -1:
                                count += 1
                                q = 1
                    if q == -1:
                        #different bot response
                        ar.append(a)
        
        a += 1
    end_time = datetime.now()
    print("-------------------------------------------------------------------------------------------------")
    print("Number of correctly answered questions: ", count)
    print("Number of wrongly answered questions: ", len(ar))
    print("Wrongly answered questions: ", ar)
    print("Time taken for testing ", len(raw_questions)," questions (in seconds): ", (end_time-start_time).total_seconds())


# ### Test twisted questions chatbot

# In[35]:


# The method tests the chatbot for given twisted questions
def TwistChatbot(raw_questions,raw_responses,twist_ques,twist_res,qn_intents,res_intents,POS_Tag_len_to_match,word_to_qn_intents_dict,word_to_res_intents_dict,questionlist,responselist,match_threshold,min_lim,max_lim,syn_dict):
    print("ROBO: My name is ISRO Robo. I will answer your general queries about ISRO. Let us chat. If you want to exit, type Bye!")
    ar = []
    a = 0
    count = 0
    start_time = datetime.now()
    while(a < len(twist_ques)):
        print("-------------------------------------------------------------------------------------------------")
        user_response = twist_ques[a]
        print("USER: ", user_response)

        #spell check
        user_response = spell_check(user_response)
        #print("SPELL: " + user_response)

        if(user_response.lower() == 'bye' or user_response.lower() == 'no'):
            print("ROBO: Bye! take care...")
            break
        elif (user_response == 'thanks' or user_response == 'thank you' ):
            print("ROBO: You are welcome... Do you have any other question?")
        else:
            grt = greeting(user_response)
            #greeting
            if(grt != None):
                print("ROBO: ", grt)
                count += 1
            elif 'fullform' in user_response or 'full form' in user_response:
                #full form check
                response = find_fullform(user_response)
                if not response:
                    print("ROBO: " + "Sorry I don't have enough information to answer you")
                    ar.append(a)
                else:
                    print("ROBO: " + response)
                    count += 1
            else:
                #abbreviation replace
                user_response = abbreviation_check(user_response)
                #print("ABB: " + user_response)

                #entity extraction
                ent = entity_extraction(user_response)

                #find for matching entities
                #bot_response
                bot_response_list = getClosestBotResponse(user_response,qn_intents,res_intents,POS_Tag_len_to_match,word_to_qn_intents_dict,word_to_res_intents_dict,match_threshold,min_lim,max_lim,syn_dict,ent)
                if not bot_response_list:
                    print("ROBO: " + "Sorry I don't have enough information to answer you")
                    ar.append(a)
                elif type(bot_response_list) == type(''):
                    #one word description
                    print("ROBO: " + bot_response_list)
                    count += 1
                elif len(bot_response_list) == 1:
                    print("ROBO: " + raw_responses[bot_response_list[0][0][0]])
                    if raw_responses[bot_response_list[0][0][0]] == twist_res[a]:
                        count += 1
                    else:
                        #different bot response
                        ar.append(a)
                elif len(bot_response_list) > 1:
                    print("ROBO: " + "I am afraid that my response may not be precise. I will provide you relevant information.") 
                    q = -1
                    for k in range(0,len(bot_response_list)):
                        print(raw_responses[bot_response_list[k][0][0]])
                        if raw_responses[bot_response_list[k][0][0]] == twist_res[a]:
                            if q == -1:
                                count += 1
                                q = 1
                    if q == -1:
                        #different bot response
                        ar.append(a)
        
        a += 1

    end_time = datetime.now()
    print("-------------------------------------------------------------------------------------------------")
    print("Number of correctly answered questions: ", count)
    print("Number of wrongly answered questions: ", len(ar))
    print("Wrongly answered questions: ", ar)
    print("Time taken for testing ", len(twist_ques)," questions (in seconds): ", (end_time-start_time).total_seconds())


# ### Start chatbot

# In[36]:


# The method starts the chat and retrieves most appropriate Chatbots response
def StartChatbot(usertext,raw_questions,raw_responses,qn_intents,res_intents,POS_Tag_len_to_match,word_to_qn_intents_dict,word_to_res_intents_dict,questionlist,responselist,match_threshold,min_lim,max_lim,syn_dict):
    print("ROBO: My name is ISRO Robo. I will answer your general queries about ISRO. Let us chat. If you want to exit, type Bye!")
    
    print("-------------------------------------------------------------------------------------------------")

    # user_response = input("User: ")
    user_response =usertext
    start_time = datetime.now()

    #spell check
    user_response = spell_check(user_response)
    #print("SPELL: " + user_response)

    if(user_response.lower() == 'bye' or user_response.lower() == 'no'):
        print("ROBO: Bye! take care...")
        return "Bye! take care..." 
        
    elif (user_response == 'thanks' or user_response == 'thank you' ):
        print("ROBO: You are welcome... Do you have any other question?")
        return "You are welcome... Do you have any other question?"
    else:
        grt = greeting(user_response)
        if (grt != None):
            print("ROBO: ", grt)
            return grt
        elif 'fullform' in user_response or 'full form' in user_response:
            #full form check
            response = find_fullform(user_response)
            if not response:
                print("ROBO: " + "Sorry I don't have enough information to answer you")
                print("-- UNKNOWN FULL FORM --")
                return "ROBO: " + "Sorry I don't have enough information to answer you"+ " -- UNKNOWN FULL FORM --"
            else:
                print("ROBO: " + response)
                return response 
        else:
            #abbreviation check
            user_response = abbreviation_check(user_response)
            #print("ABB: " + user_response)

            #entity extraction
            ent = entity_extraction(user_response)

            #find for matching entities
            #bot_response
            # global res
            res=[]
            bot_response_list = getClosestBotResponse(user_response,qn_intents,res_intents,POS_Tag_len_to_match,word_to_qn_intents_dict,word_to_res_intents_dict,match_threshold,min_lim,max_lim,syn_dict,ent)
            if not bot_response_list:
                print("ROBO: " + "Sorry I don't have enough information to answer you")
                print("-- NO BOT RESPONSE --")
                return "Sorry I don't have enough information to answer you"
            elif type(bot_response_list) == type(''):
                print("ROBO: " + bot_response_list)
                return bot_response_list 
            elif len(bot_response_list) == 1:
                print("ROBO: " + raw_responses[bot_response_list[0][0][0]])
                return raw_responses[bot_response_list[0][0][0]]
            elif len(bot_response_list) > 1:
                respi=""
                print("ROBO: " + "I am afraid that my response may not be precise. I will provide you relevant information. \n")
                for k in range(0,len(bot_response_list)):
                    print(raw_responses[bot_response_list[k][0][0]])

                    res.append(raw_responses[bot_response_list[k][0][0]])
                    respi=respi+"."+raw_responses[bot_response_list[k][0][0]]
                return "I am afraid that my response may not be precise. I will provide you relevant information. \n"+ respi

    end_time = datetime.now()
    print("Time (in seconds): ", (end_time-start_time).total_seconds())


# ### Build chatbot

# In[37]:


def BuildChatbot(original_corpus):
    print("BUILDING CHATBOT...")
    print("Takes some time... Please wait!")
    
    #Abbreviation detection
    if os.path.exists("abbreviated_corpus_s.csv") == False or os.path.exists("abbreviations_s.pickle") == False:
        print("Abbreviation files are being created...")
        time1 = datetime.now()
        abb_det = Abbreviations_Detection()
        abb_det.Abbreviations(original_corpus)
        abb_det.Update_abbreviations('abbreviations_s.csv', original_corpus)
        time2 = datetime.now()
        print("Abbreviation files created in (seconds): ", (time2-time1).total_seconds())
    
    #questions and intents
    if os.path.exists("questions_s.pickle") == False or os.path.exists("responses_s.pickle") == False:
        print("Raw corpus files are being created...")
        time1 = datetime.now()
        raw_questions, raw_responses = getQuestionseAndResponseList(original_corpus,'questions','responses')
        create_pickle("questions_s.pickle", raw_questions)
        create_pickle("responses_s.pickle", raw_responses)
        time2 = datetime.now()
        print("Raw corpus files created in (seconds): ", (time2-time1).total_seconds())
    
    if os.path.exists("abb_questions_s.pickle") == False or os.path.exists("abb_responses_s.pickle") == False:
        print("Abbreviated corpus files are being created...")
        time1 = datetime.now()
        questionlist, responselist = getQuestionseAndResponseList("abbreviated_corpus_s.csv",'questions','responses')
        create_pickle("abb_questions_s.pickle", questionlist)
        create_pickle("abb_responses_s.pickle", responselist)
        time2 = datetime.now()
        print("Abbreviated corpus files created in (seconds): ", (time2-time1).total_seconds())
    
    if os.path.exists("questions_intents_s.pickle") == False or os.path.exists("responses_intents_s.pickle") == False or os.path.exists("word_to_qn_intents_dict_s.pickle") == False or os.path.exists("word_to_res_intents_dict_s.pickle") == False:
        time1 = datetime.now()
        print("Intents files are being created...")
        questionlist = pd.read_pickle("abb_questions_s.pickle")
        responselist = pd.read_pickle("abb_responses_s.pickle")
        qn_intents = getIntents(questionlist)
        res_intents = getIntents(responselist)
        create_pickle("questions_intents_s.pickle", qn_intents)
        create_pickle("responses_intents_s.pickle", res_intents)
        syn_dict = {}
        word_to_qn_intents_dict = getWordTOIntentsDict(qn_intents,syn_dict)
        word_to_res_intents_dict = getWordTOIntentsDict(res_intents,syn_dict)
        create_pickle("syn_dict_s.pickle", syn_dict)
        create_pickle("word_to_qn_intents_dict_s.pickle", word_to_qn_intents_dict)
        create_pickle("word_to_res_intents_dict_s.pickle", word_to_res_intents_dict)
        time2 = datetime.now()
        print("Intents files created in (seconds): ", (time2-time1).total_seconds())


# In[38]:


def create_files():
    #Entities and labels
    if os.path.exists("entities_and_labels_responses_dictionary_s.pickle") == False or os.path.exists("entities_and_labels_questions_dictionary_s.pickle") == False:
        print("Entity labels are being created...")
        time1 = datetime.now()
        generate_entities_and_labels_dictionary('abbreviated_corpus_s.csv')
        time2 = datetime.now()
        print("Entity labels created in (seconds): ", (time2-time1).total_seconds())
    
    #Phrases
    if os.path.exists("phrase_corpus_s.csv") == False or os.path.exists("phrases_s.pickle") == False or os.path.exists("series_s.pickle") == False:
        print("Phrases files are being created...")
        time1 = datetime.now()
        with open('entities_and_labels_questions_dictionary_s.pickle', 'rb') as f: 
            entities = pickle.load(f)
        with open('entities_and_labels_responses_dictionary_s.pickle', 'rb') as f: 
            en = pickle.load(f)
        for i in en.keys():
            if i not in entities.keys():
                entities[i]=en[i]    
        P = Phrase_Collection()
        sp,series = P.Phrases_with_hyphen('abbreviated_corpus_s.csv')
        P.Phrase_collection('abbreviated_corpus_s.csv',entities,sp,scoring='npmi')
        time2 = datetime.now()
        print("Phrases files created in (seconds): ", (time2-time1).total_seconds())
    
    #Synonyms generation
    if os.path.exists("distinct_words_s.csv") == False or os.path.exists("word_synonymlist_s.pickle") == False:
        print("Synonyms files are being created...")
        time1 = datetime.now()
        generate_distinct_words(original_corpus)
        generate_synonyms('distinct_words_s.csv')
        time2 = datetime.now()
        print("Synonyms files created in (seconds): ", (time2-time1).total_seconds())
    
    #One word descriptions
    if os.path.exists("one_word_descriptions_s.pickle") == False:
        print("One word descriptions are being created...")
        time1 = datetime.now()
        nouns_list = get_nouns_list(qn_intents)
        nouns_list.extend(get_nouns_list(res_intents))
        definations_s1=definations_s('abbreviated_corpus_s.csv','series_s.pickle', nouns_list)
        time2 = datetime.now()
        print("One word descriptions created in (seconds): ", (time2-time1).total_seconds())

    #Entity extraction
    if os.path.exists("entities_mapped_to_questions_dictionary_s.pickle") == False or os.path.exists("entities_mapped_to_responses_dictionary_s.pickle") == False:
        print("Entity files are being created...")
        time1 = datetime.now()
        generate_entities_mapped_to_corpus_dictionary('abbreviated_corpus_s.csv')
        time2 = datetime.now()
        print("Entity files created in (seconds): ", (time2-time1).total_seconds())

    #Corpus dictionary
    if os.path.exists("corpus_s.txt") == False or os.path.exists("words_set_s.pickle") == False:
        print("Corpus dictionary is being created...")
        time1 = datetime.now()
        build_corpus_dictionary(original_corpus)
        time2 = datetime.now()
        print("Corpus dictionary created in (seconds): ", (time2-time1).total_seconds())
        
    print("Required files created")


# ### Load generated files

# In[39]:



start_time = datetime.now()

# original_corpus = r'C:\Users\Admin\Documents\2020501072\DS_2ndyear\Final chatbot files\QuestionsResponsesISRO_s.csv'
# twisted_questions = r'C:\Users\Admin\Documents\2020501072\DS_2ndyear\Final chatbot files\test_questions_s.csv'
# BuildChatbot(original_corpus)


def own_func():
    global original_corpus,twisted_questions
    original_corpus = "QuestionsResponsesISRO.csv"
    twisted_questions = "test_questions.csv"
    
    print(os.path.abspath(os.getcwd()))
    # original_corpus = oc
    # twisted_questions = tq
    BuildChatbot(original_corpus)

#questions and intents
# self.raw_questions =  pd.read_pickle("questions_s.pickle")
# self.raw_responses =  pd.read_pickle("responses_s.pickle")
# self.questionlist =  pd.read_pickle("abb_questions_s.pickle")
# self.responselist =  pd.read_pickle("abb_responses_s.pickle")
# self.qn_intents =  pd.read_pickle("questions_intents_s.pickle")
# self.res_intents =  pd.read_pickle("responses_intents_s.pickle")
# self.syn_dict =  pd.read_pickle("syn_dict_s.pickle")
# self.word_to_qn_intents_dict =  pd.read_pickle("word_to_qn_intents_dict_s.pickle")
# self.word_to_res_intents_dict =  pd.read_pickle("word_to_res_intents_dict_s.pickle")

    
    global raw_questions,raw_responses,questionlist,responselist,qn_intents,res_intents,syn_dict,word_to_qn_intents_dict,word_to_res_intents_dict
    raw_questions =  pd.read_pickle("questions_s.pickle")
    raw_responses =  pd.read_pickle("responses_s.pickle")
    questionlist =  pd.read_pickle("abb_questions_s.pickle")
    responselist =  pd.read_pickle("abb_responses_s.pickle")
    qn_intents =  pd.read_pickle("questions_intents_s.pickle")
    res_intents =  pd.read_pickle("responses_intents_s.pickle")
    syn_dict =  pd.read_pickle("syn_dict_s.pickle")
    word_to_qn_intents_dict =  pd.read_pickle("word_to_qn_intents_dict_s.pickle")
    word_to_res_intents_dict =  pd.read_pickle("word_to_res_intents_dict_s.pickle")

    create_files()

    #abbreviations
    global abbreviations
    abbreviations = pd.read_pickle('abbreviations_s.pickle')
    global abbreviations_dict
    abbreviations_dict = {}
    for i in abbreviations:
        for j in abbreviations[i]:
            abbreviations_dict[j] = abbreviations[i][j].upper()

    #phrases
    global sp,sin_word_def
    sp = pd.read_pickle("phrases_s.pickle")

    #one word descriptions
    sin_word_def = pd.read_pickle("one_word_descriptions_s.pickle")

    #entity extraction
    global arr,arr1,corpus_dict,standard_dict,corpus_dic
    arr = pd.read_pickle("entities_mapped_to_questions_dictionary_s.pickle")
    arr1 = pd.read_pickle("entities_mapped_to_responses_dictionary_s.pickle")

    #spell check    
    corpus_dict = enchant.PyPWL("corpus_s.txt")
    standard_dict = enchant.Dict("en_US")
    corpus_dic = pd.read_pickle("words_set_s.pickle")

    end_time = datetime.now()
    print("Time (in minutes): ", (end_time-start_time).total_seconds()/60)
    global ques,res
    ques = []
    for i in raw_questions:
        if i not in ques:
            ques.append(i)

    values = np.array(raw_questions)
    qindex = []
    for i in range(len(ques)):
        search = ques[i]
        k = np.where(values == search)[0]
        qindex.append(k)

    res = []
    for i in qindex:
        rs = []
        for j in i:
            rs.append(raw_responses[j])
        res.append(rs)
        
    csv = pd.read_csv(twisted_questions)
    # csv = pd.read_csv(tq)
    twist_ques = list(csv['questions'])
    twist_res = list(csv['responses'])


    # ### Testing

    # In[40]:


    #testing corpus questions
    TestChatbot(raw_questions,raw_responses,qn_intents,res_intents,2,word_to_qn_intents_dict,word_to_res_intents_dict,questionlist,responselist,0.5,2,5,syn_dict)


    # In[41]:


    #testing twisted questions
    TwistChatbot(raw_questions,raw_responses,twist_ques,twist_res,qn_intents,res_intents,2,word_to_qn_intents_dict,word_to_res_intents_dict,questionlist,responselist,0.5,2,5,syn_dict)


    # ### Start

    # In[43]:

    # def startbot(self,usertext):
def startbot(usertext):
    return StartChatbot(usertext,raw_questions,raw_responses,qn_intents,res_intents,2,word_to_qn_intents_dict,word_to_res_intents_dict,questionlist,responselist,0.5,2,5,syn_dict)

    # StartChatbot(raw_questions,raw_responses,qn_intents,res_intents,2,word_to_qn_intents_dict,word_to_res_intents_dict,questionlist,responselist,0.5,2,5,syn_dict)

# In[ ]:





# In[ ]:

# original_corpus = r'C:\Users\Admin\Documents\2020501072\DS_2ndyear\Final chatbot files\QuestionsResponsesISRO_s.csv'
# twisted_questions = r'C:\Users\Admin\Documents\2020501072\DS_2ndyear\Final chatbot files\test_questions_s.csv'

# own_func()




     