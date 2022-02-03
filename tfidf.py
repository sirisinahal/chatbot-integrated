#!/usr/bin/env python
# coding: utf-8

# In[15]:


import re #used in removing punctuations
import random #used to pick a random response
import nltk
import nltk
# nltk.download()
from nltk.corpus import stopwords #used in stopwords removal
from nltk.tokenize import word_tokenize #used in tokenization
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer #used in lemmatization
lemmatizer=WordNetLemmatizer()
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
from itertools import chain
import math #used in idf computation
import numpy as np #used to convert list to numpy array
from scipy import sparse #used in converting vector to sparse matrix
import pandas as pd #used to fetch data
from sklearn.metrics.pairwise import cosine_similarity #used to find similarity with documents
import enchant #used in spellchecking
import sys
from spellchecker import SpellChecker
import pickle
import spacy
from spacy import displacy
from num2words import num2words
from datetime import datetime
import os
import csv
from csv import reader
import random
import codecs

# # Explore the data


def extract_corpus_file(original_corpus):
    dataset=pd.read_csv(original_corpus) #corpus after replacing abbreviations and phrases
    questions=list(dataset['questions'])
    responses=list(dataset['responses'])
    
    pickle_create('questions_t.pickle',questions)
    pickle_create('responses_t.pickle',responses)
    
    return questions,responses


def pickle_create(filename,data):
    with open(filename, 'wb') as handle:
        pickle.dump(data,handle, protocol=2)


# # Abbreviations

class Abbreviations_Detection():
        
    def replace_abb(self,abbreviations,user_query):
        import re
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
        import pickle
        import csv
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
         
        with open('abb_corpus_t.csv','w',newline='',encoding='utf-8') as file:
            writer=csv.writer(file)
            writer.writerow(['questions','responses'])
            for i in range(len(questions)):
                writer.writerow([questions[i],responses[i]])
        
        
        filename1 = 'abbreviations_t.pickle' 
        outfile = open(filename1,'wb') 
        pickle.dump(abbreviations1,outfile) 
        outfile.close()
        
        
        return filename1
    
    
    def Abbreviations(self,filename):
        import csv
        import requests
        import pickle
        from bs4 import BeautifulSoup
        import re
        import spacy
        from scispacy.abbreviation import AbbreviationDetector
        from spacy.language import Language
        import wikipedia
        
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
        
        with open('abb_corpus_t.csv','w',newline='',encoding='utf-8') as file:
            writer=csv.writer(file)
            writer.writerow(['questions','responses'])
            for i in range(len(questions)):
                writer.writerow([questions[i],responses[i]])
        with open('abbreviations_t.csv','w',newline='',encoding='utf-8') as file:
            writer=csv.writer(file)
            writer.writerow(['full form','short form'])
            for i in abbreviations.keys():
                writer.writerow([i,abbreviations[i]])
                
        filename1 = 'abbreviations_t.pickle' 
        outfile = open(filename1,'wb') 
        pickle.dump(abbreviations1,outfile) 
        outfile.close()
        
        return 'abb_corpus_t.csv',filename1,'abbreviations_t.csv'
        

def abbreviation_check(abbreviations,user_query):
        import re
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


def create_abbreviations_dict(abb_check):
    abbreviations={}
    for i in abb_check.values():
        for j in i.keys():
            abbreviations[j]=i[j]
            
    pickle_create('detected_abbreviations_t.pickle',abbreviations)
    
    abb=pd.read_pickle('detected_abbreviations_t.pickle')
    shortform=list(abb.values())
    fullform=list(abb.keys())

    fullform=remove_p(fullform) #remove special characters other than alphabet and numbers
    shortform=remove_p(shortform)

    abbreviations={}
    for i in range(len(shortform)):
        abbreviations[fullform[i]]=shortform[i]
    
    pickle_create('detected_abbreviations_t.pickle',abbreviations)


def remove_p(a): #remove any special characters
    for i in range(len(a)):
        a[i]=a[i].replace('-',' ')
        a[i]=re.sub(r'[^\w\s]','',a[i])
    return a


def create_abbtokens(fullforms): #create a list of tokens from all the abbreviations
    global abbtokens
    abbtokens=[]
    phrases_list=pd.read_pickle('detected_phrases_t.pickle')
    for i in range(len(fullforms)):
        if i not in phrases_list:
            abbtokens.extend([nltk.word_tokenize(fullforms[i])])
        else:
            abbtokens.extend(fullforms[i])
    return abbtokens


def create_distinct_abb_tokens(fullform):
    abbtokens=create_abbtokens(fullform) #create tokens for abbreviations
    
    distinct_abb=[]
    for i in abbtokens:
        for j in i:
            if j not in distinct_abb:
                distinct_abb.append(j)
    pickle_create('distinct_abb_t.pickle',distinct_abb)
    return distinct_abb


def find_fullform(tokens): #given a question in the form of 'what is the fullform of xxxx', return its fullform
    for token in tokens:
        #print(token)
        if(token!='fullform'):
            abbrev=token.upper()
    return fullform[shortform.index(abbrev)]


# # Phrase collection


class Phrase_Collection():
    
    def Update_phrases(self,filename,filename1):
        import pickle
        import csv
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
            

        with open('phrase_corrected_corpus_t.csv','w',newline='',encoding='utf-8')as file:
            writer=csv.writer(file)
            writer.writerow(['questions','responses'])
            for i in range(len(questions)):
                writer.writerow([questions[i],responses[i]])
        
        filename1 = 'phrase_check_t.pickle' 
        outfile = open(filename1,'wb') 
        pickle.dump(phrases1,outfile) 
        outfile.close()
        
        
        return filename1
    
    def Phrases_with_hyphen(self,filename):
        import csv
        import pickle
        import re

        
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
        
        filename1='series_t.pickle'
        outfile=open(filename1,'wb')
        pickle.dump(series,outfile)
        outfile.close()
        
        with open('phrase_t.csv','w',newline='',encoding='utf-8')as file:
            writer=csv.writer(file)
            writer.writerow(['phrases'])
            for i in set(co):
                writer.writerow([i])
        
        
        return sp,filename1
    
    
    def Phrase_collection(self,filename,entities,sp={},scoring='default',min_count=4,threshold=0.88):
        import csv
        import pickle
        import re
        from gensim.test.utils import datapath
        from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
        from nltk.tokenize import word_tokenize
        import nltk
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
            

        with open('phrase_corrected_corpus_t.csv','w',newline='',encoding='utf-8')as file:
            writer=csv.writer(file)
            writer.writerow(['questions','responses'])
            for i in range(len(questions)):
                writer.writerow([questions[i],responses[i]])
                
        with open('phrase_t.csv','a',newline='',encoding='utf-8')as file:
            writer=csv.writer(file)
            writer.writerow(['phrases'])
            for i in phrase:
                writer.writerow([i])

        filename1='phrase_check_t.pickle'
        outfile=open(filename1,'wb')
        pickle.dump(sp,outfile)
        outfile.close()

        return 'phrase_corrected_corpus_t.csv','phrase_check_t.pickle','phrase_t.csv'
    
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


def generate_phrases_list():
    phrase_check=pd.read_pickle('phrase_check_t.pickle')
    
    phrases_list=[]
    for i in phrase_check.values():
        for j in i.values():
            if j.lower() not in phrases_list:
                phrases_list.append(j.lower())          
    
    pickle_create('detected_phrases_t.pickle',phrases_list)


def generate_phrase_to_word_dict(phrases_list):
    phrases_dict={}
    for i in range(len(phrases_list)):
        phrase_key=phrases_list[i]
        phrases_value=remove_u(phrases_list[i])
        phrases_dict[phrase_key]=phrases_value
    
    pickle_create('phrase_to_word_dictionary_t.pickle',phrases_dict)


def remove_u(sentence):
    sentence=re.sub(r'\_',' ',sentence)
    return sentence


def phrases_replace(sentences): # call replace_hyphen(user_query,di)
    phrase_check=pd.read_pickle('phrase_check_t.pickle')
    for i in sentences.split():
        if '-' in i:
            y=i.split('-')
            if y[0] in phrase_check.keys():
                for j in phrase_check[y[0]].keys():
                    sentences=sentences.replace(j,phrase_check[y[0]][j])
        else:
            if i in phrase_check.keys():
                for j in phrase_check[i].keys():
                    sentences=sentences.replace(j,phrase_check[i][j])
    return sentences


def change_phrase_to_word(tokens):
    for i in range(len(tokens)):
        if '_' in tokens[i]:
            tokens[i]=re.sub(r'\_',' ',tokens[i])
    return tokens


# # Entity extraction


def extract_entities(corpus_file):
    nlp = spacy.load("en_core_web_sm") #load spacy model
    arr = []     #question_entities
    arr1 = []    #response_entities
    i = 0

    with open(corpus_file, 'r', encoding = "utf8") as read_obj:
        csv_reader = list(reader(read_obj))
        for row in csv_reader:
            if(row[0][-1].isalnum() == False):
                doc = nlp(row[0][:-1])
            else:
                doc = nlp(row[0])
            arr.append({})
            for ent in doc.ents:
                arr[i][str(ent)] = ent.label_

            if(row[1][-1].isalnum() == False):
                doc = nlp(row[1][:-1])
            else:
                doc = nlp(row[1])
            arr1.append({})
            for ent in doc.ents:
                arr1[i][str(ent)] = ent.label_
            i = i + 1
    
    arr.remove(arr[0])
    arr1.remove(arr1[0])
    
    return arr,arr1

def generate_entity_dictionary(array):
    dis_ent_dict = {}
    for i in range(0, len(array)):
        dic = array[i]
        for j in dic:
            k = j.lower()
            if k not in dis_ent_dict:
                dis_ent_dict[k] = {i}
            else:
                dis_ent_dict[k].add(i)
            
    return dis_ent_dict

def create_pickle(file, data):
    open_file = open(file, 'wb')
    pickle.dump(data, open_file)
    open_file.close()

def generate_entities(corpus_file):
    arr,arr1=extract_entities(corpus_file)

    dis_ent_dict_que=generate_entity_dictionary(arr)
    dis_ent_dict_res=generate_entity_dictionary(arr1)

    create_pickle('question_entities_to_index_t.pickle', dis_ent_dict_que)
    create_pickle('response_entities_to_index_t.pickle', dis_ent_dict_res)


#entity extraction
def get_entities2(user_input):
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
        
    #print("ENT: ", shared_items_dic)
    
    if(len(shared_items_dic) == 0):
        return shared_items_dic     
        #bot_response
    else:
        return shared_items_dic


# # Synonym generation

# generating intents,nouns,distinct words

def tokenize_words_pos(text):
    tokenized_words=[]
    stopwrds=stopwords.words('english')
    #stopwrds.remove('no')                  #include are exclude words according to the domain
    stopwrds.remove('not')
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer() 
    for i in text:
        #i=i.lower()
        wrd_token=nltk.pos_tag(tokenizer.tokenize(i))  
        wrd_token=[(word[0].lower(),word[1]) for word in wrd_token]       #converting to lower case
        wrd_token=[(lemmatizer.lemmatize(lemmatizer.lemmatize(word[0]),pos='v'),word[1]) for word in wrd_token if word[0] not in stopwrds] #stopword removal and lemmatization
        #if not wrd_token:
            #wrd_token=tokenizer.tokenize(i)
        
        tokenized_words.append(wrd_token)
        
    return tokenized_words


def derive_intents(intents_pos):
    ans_intents_derive=[]
    for j in intents_pos:
        lt=[]
        for k in j:
            lt.append(k[0])
            lt.sort()
        ans_intents_derive.append(lt)
    return ans_intents_derive
    

def make_list(dt):
    lq=[]
    for i in dt:
        for j in i:
            if j not in lq:
                lq.append(j)
    return lq

def get_nouns_list():
    que_list=pd.read_pickle("questions2_t.pickle")
    ans_list=pd.read_pickle("responses2_t.pickle")
    
    que_intents_pos=pd.read_pickle('que_intents_pos_t.pickle')
    ans_intents_pos=pd.read_pickle('ans_intents_pos_t.pickle')
    
    corpus_intents_pos=que_intents_pos+ans_intents_pos
    lt=['NNP']
    nouns_list=[]
    for i in corpus_intents_pos:
            for k in i:
                if k[1] in lt and k[0].lower() not in stopwords.words('english'):
                    if k[0] not in nouns_list:
                        nouns_list.append(k[0])
    
    pickle_create("nouns_list_t.pickle",nouns_list)
    return nouns_list

def get_def_list(intents):
    def_list=[]
    for i in intents:
        if len(i)==1:
            def_list.append(i[0])
    return def_list


def get_intents():
    phr=pd.read_csv('phrase_corrected_corpus_t.csv')
    ques=list(phr['questions'])
    resps=list(phr['responses'])
    pickle_create('questions2_t.pickle',ques)
    pickle_create('responses2_t.pickle',resps)
    
    que_list=pd.read_pickle("questions2_t.pickle")
    ans_list=pd.read_pickle("responses2_t.pickle")
    que_intents_pos=tokenize_words_pos(que_list)
    ans_intents_pos=tokenize_words_pos(ans_list)
    
    que_intents=derive_intents(que_intents_pos)
    ans_intents=derive_intents(ans_intents_pos)
    
    pickle_create("que_intents_t.pickle",que_intents)
    pickle_create("ans_intents_t.pickle",ans_intents)
    
    pickle_create("que_intents_pos_t.pickle",que_intents_pos)
    pickle_create("ans_intents_pos_t.pickle",ans_intents_pos)
    
    que_distinct_words=make_list(que_intents)
    ans_distinct_words=make_list(ans_intents)
    
    corpus_distinct_words=que_distinct_words+ans_distinct_words
    
    pickle_create("que_distinct_words_t.pickle",que_distinct_words)
    pickle_create("ans_distinct_words_t.pickle",ans_distinct_words)
    pickle_create("corpus_distinct_words_t.pickle",corpus_distinct_words)
    
    q=pd.read_pickle('que_intents_t.pickle')
    r=pd.read_pickle('ans_intents_t.pickle')
    
    d_list1=get_def_list(q)
    pickle_create('que_def_t.pickle',d_list1)
    
    d_list2=get_def_list(r)
    pickle_create('res_def_t.pickle',d_list2)

    
# Generating Synonyms

def getHyperNyms(word):
    word_hypernyms=set()
    syns=wordnet.synsets(word)
    for i,j in enumerate(syns):
        [word_hypernyms.add(elt) for elt in (list(chain(*[l.lemma_names() for l in j.hypernyms()])))]
    return word_hypernyms 

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

def get_synonyms(wrd,pos):
    lt1=[]
    fl=[]
    [lt1.append(lem.name().lower()) for syn in wordnet.synsets(wrd,pos) for lem in syn.lemmas()]
    [lt1.append(hypo.lower()) for hypo in getHyperNyms(wrd)]
    lt1
    for m in lt1:
        if (getPathSimilarity(wrd,m)==1):
            #print(j,m)
            if m not in fl:
                fl.append(m)
    return fl


def corpus_dict_pos(que_tokens,que_distinct_words):
    nd={}
    nb=len(que_tokens)
    mk=[]
    fl=[]
    dp=[]
    
    for i in range(0,nb): 
        for k1 in que_tokens[i]:
            fl=[]
            j=k1[0]  #getting the word
            pos_tag=k1[1][0].lower()    #getting the pos tag of word
            if k1[0] not in mk:
                mk.append(k1[0])           #finding the synonym of the word if not seen before
                if pos_tag=='j':
                    pos_tag='a'
                if pos_tag  in ['n','a','r','v','s']:  #finding the synonyms with the following pos tags due wordnet constraint
                    lt1=get_synonyms(j,pos_tag)        #getting the synonyms
                    if len(lt1)==0:                    #if the synonyms list is empty appending the same word
                        nd[j]=[j]
                    else:
                        fl=[]
                        for m in lt1:
                                if m not in fl and m not in dp:
                                    if (m in que_distinct_words):       #if the word is not a corpus word condition to avoid conflict
                                        mk.append(m)
                                    dp.append(m)
                                    fl.append(m)
                                    
                                    
                        if len(fl)==0:       #if the filtration leads to empty list appending the list with the corpus word itself
                            fl.append(j)
                            
                        nd[j]=fl 

                elif pos_tag=='c' and j.isdigit():      #finding the full word of the numbers
                    fl=[]
                    fl.append(num2words(j))
                    nd[j]=fl
    return nd

def get_phrases_list(syn_dict):
    lt=[]
    lt1=[]
    for i in list(syn_dict.values()):
        for j in i:
            if '_' in j:
                lt1.append(j)
                
    for i in list(syn_dict):
        if '_' in i:
            lt1.append(i)
            
    return lt1


def get_words_synonyms():
    doc=[]
    doc.extend(pd.read_pickle('que_intents_pos_t.pickle'))
    doc.extend(pd.read_pickle('ans_intents_pos_t.pickle'))
    pickle_create('doc_t.pickle',doc)
    doc=pd.read_pickle('doc_t.pickle')
    
    corpus_dist=pd.read_pickle('corpus_distinct_words_t.pickle')
    corpus_dist=list(set(corpus_dist))
    
    que_syn_dict=corpus_dict_pos(doc,corpus_dist)
    phrases_list=get_phrases_list(que_syn_dict)
    
    pickle_create("syn_list_t.pickle", que_syn_dict)
    pickle_create("phrases_list_t.pickle",phrases_list)
    
    syns_dict=pd.read_pickle("syn_list_t.pickle")
    #words_synonyms_list=make_list_dict(syns_dict)
    syns_dict_rep=duplicating_tuples(syns_dict)
    #corpus_dictionary_index=synonyms_indexing(syns_dict_rep,words_synonyms_list)
    corpus_dictionary_index=synonyms_indexing(syns_dict_rep,syns_dict)
    pickle_create("syn_ind_t.pickle",corpus_dictionary_index)
    
    
def duplicating_tuples(dt):
    nd={}
    for i,j in zip(dt.keys(),dt.values()):
        nd[i]=j
        for k in j:
            nd[k]=j
    return nd



def synonyms_indexing(dt,dt1):
    rk=dt.keys()
    lt=list(dt1.values())
    tf={}
    dp=[]
    for r in rk:
        ls=dt[r]
        ind=lt.index(ls)
        tf[r]=ind
    return tf

def find_distinct_words():
    distinct_words=list(pd.read_pickle('syn_list_t.pickle').keys())
    obj=pd.DataFrame(distinct_words)
    obj.to_csv('distinct_words_t.csv',index=False,header=['words'])


# # Generate TF-IDF vectors for corpus


def find_tf(documents,distinct_words): #compute term frequency (TF) for each term in the document
    #distinct_words=list(pd.read_pickle('syn_list_t.pickle').keys())
    syn_list=list(pd.read_pickle('syn_list_t.pickle').values())
    tfcount=[[0 for word in distinct_words] for doc in documents] 
    for docidx in range(len(documents)):
        for synidx in range(len(distinct_words)):
            if distinct_words[synidx] in documents[docidx]:
                tfcount[docidx][synidx]=documents[docidx].count(distinct_words[synidx])
            else:
                temp=syn_list[synidx]
                for i in temp:
                    if i in documents[docidx]:
                        tfcount[docidx][synidx]=documents[docidx].count(i)
                    else:
                        continue
        #tfcount[docidx]=[i/max(tfcount[docidx]) for i in tfcount[docidx]] #normalised frequency
    tfcount=np.array(tfcount)
    #print(tfcount)
    return tfcount

def find_idf(documents,distinct_words): #compute document frequency for each term
    df=[0 for word in distinct_words] 
    for term in distinct_words:
        count=0
        for doc in documents:
            if term in doc:
                count+=1
        df[distinct_words.index(term)]=count
    for termidx in range(len(df)): #compute Inverse Document Frequency for each term
        df[termidx]=(math.log2(len(documents))/(1+df[termidx])) # idf= log(N/(n+1))
    df=np.array(df)
    #print(df)
    return df

def find_tfidf(documents,distinct_words,tf,idf): #computer TF-IDF values by multiplying tf value with idf value 
    tfidf=[[0 for word in distinct_words] for doc in documents] 
    for docidx in range(len(documents)):
        tfidf[docidx]=tf[docidx]*idf
    tfidf_sparse=sparse.csr_matrix(np.array(tfidf))
    #print(tfidf_sparse) #display the final tfidf values
    return tfidf_sparse

def generate_tfidf_vectors(que_documents_file,res_documents_file,distinct_words_file):
    que_documents=pd.read_pickle('que_intents_t.pickle')
    res_documents=pd.read_pickle('ans_intents_t.pickle')
    
    documents=pd.read_pickle('doc_t.pickle')
    
    distinct_words=list(pd.read_csv('distinct_words_t.csv')['words'])
    
    idf=find_idf(documents,distinct_words) #find idf for corpus
    
    que_tf=find_tf(que_documents,distinct_words) #find tf for questions
    que_tfidf=find_tfidf(que_documents,distinct_words,que_tf,idf) #find tf-idf for questions
    
    res_tf=find_tf(res_documents,distinct_words) #find tf for responses
    res_tfidf=find_tfidf(res_documents,distinct_words,res_tf,idf) #find tf-idf for responses
    
    pickle_create('idf_vector_t.pickle',idf)
    pickle_create('question_tfidf_vectors_t.pickle',que_tfidf)
    pickle_create('response_tfidf_vectors_t.pickle',res_tfidf)


# # Single word queries

def get_definations_t(i):# i is list of nouns in caps or mixed
    import csv
    import re
    import pickle
    import wikipedia
    import nltk
        
    def get_from_web(c):
        
        try:
            results=wikipedia.search(c)
            info=wikipedia.page(results[0]).summary
        except:
            info='not available'
            pass
        return info
        
    definations_t={}
    for j in i:
        text=get_from_web(j)
        if text!='not available':
            text=text.replace('\n','')

            text=text.encode('ascii',errors='ignore').decode('ascii')
            sent_tokens=[]
            [sent_tokens.append(sent) for sent in nltk.sent_tokenize(text)]
            for k in range(2):
                try:
                    if j.lower() not in definations_t.keys():
                        definations_t[j.lower()]=sent_tokens[k]+' '
                    else:
                        definations_t[j.lower()]+=sent_tokens[k]+' '
                except:
                    pass
    filename='single_word_definations_t_t.pickle'
    output=open(filename,'wb')
    pickle.dump(definations_t,output)
    output.close()
        
        
    return definations_t

def generate_single_word_definitions():
    series=set()
    
    series1=pd.read_pickle('series_t.pickle')
    nouns=get_nouns_list()

    for i in series1:
        series.add(i)
    for i in nouns:
        series.add(i)
        
    series=list(series)
    
    que_def=pd.read_pickle('que_def_t.pickle')
    series=[i for i in series if i not in que_def]
    
    definations_t=get_definations_t(series)


def generate_query_description_dict():
    query=[i.lower() for i in query_definition.keys()]
    definition=[i for i in query_definition.values()]
    query_description={}
    for i in range(len(query)):
        query_description[query[i]]=definition[i]
    print(query_description)


# # Generate corpus dictionary


def generate_corpus_enc():
    if os.path.exists('corpus_t.txt'):
        os.remove('corpus_t.txt')
    #text file that stores the distinct words
    corpus_dict = open("corpus_t.txt", "a", encoding = 'cp1252')
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
    syn = pd.read_pickle('syn_list_t.pickle')
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
    abb = pd.read_pickle('abbreviations_t.pickle')
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
    di = pd.read_pickle('phrase_check_t.pickle')
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
    with open('single_word_definations_t_t.pickle', 'rb') as read_obj:
        pkl_reader = pickle.load(read_obj)
        for items in pkl_reader.items():
            tokenizer = nltk.RegexpTokenizer(r"\w+")
            words_list = tokenizer.tokenize(str(items))
            for word in words_list:
                if word.isnumeric() == False:
                    words_set.add(word)
                    words_set.add(word.lower())
                    
    create_pickle('words_set_t.pickle',list(words_set))

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


def spell_check(user_input): 
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    words_list = tokenizer.tokenize(user_input)
    
    for word in words_list:
        if word not in dic:
            if word.isnumeric() == False:
                if standard_dict.check(word) == False:
                    suggestions = corpus_dict.suggest(word)
                    if(len(suggestions) != 0):
                        user_input = user_input.replace(word, suggestions[0])
                    else:
                        user_input = user_input.replace(word, standard_dict.suggest(word)[0])

    return user_input


# # TF-IDF for user query


def find_user_tf(tokens):
    syn_list=list(pd.read_pickle('syn_list_t.pickle').values())
    usertf=[0 for word in distinct_words] #compute term frequency (TF) for each term in the user input
    for token in tokens:
        if token in synonyms:
            idx=syn_index[synonyms.index(token)] #find the index at which tf should be updated
            usertf[idx]+=1
        elif token in distinct_abb: #if the token appears in abbreviation-fullform eg:weapon,aperture
            for i in abbtokens:
                if token in i:
                    k=shortform[abbtokens.index(i)]
            idx=syn_index[synonyms.index(k)]
            usertf[idx]+=1
        else:
            continue
            
    #usertf=[i/max(usertf) for i in usertf] #normalised tf
    usertf=np.array(usertf)
    #print(usertf)
    return usertf


def find_user_tfidf(usertf): #computer TF-IDF values by multiplying tf value with idf value
    usertfidf=[0 for word in distinct_words]
    usertfidf=usertf*idf
    user_tfidf_sparse=sparse.csr_matrix(np.array(usertfidf))
    #print(user_tfidf_sparse)
    return user_tfidf_sparse


def find_intents(user_input):    
    intents=[]
    stopwrds=stopwords.words('english')
    stopwrds.remove('not')
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer() 

    wrd_token=nltk.pos_tag(tokenizer.tokenize(user_input))  
    wrd_token=[(word[0].lower(),word[1]) for word in wrd_token]       #converting to lower case
    wrd_token=[(lemmatizer.lemmatize(lemmatizer.lemmatize(word[0]),pos='v'),word[1]) for word in wrd_token if word[0] not in stopwrds] #stopword removal and lemmatization
    
    if wrd_token:
        intents=[i[0] for i in wrd_token]
        
    return intents


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# # Find response without using entities


def find_response(usertokens):
    #print(usertokens)
    try:
        user_tfidf=find_user_tfidf(find_user_tf(usertokens))
        #print(user_tfidf)
    except:
        return "Sorry! I couldn't understand u! Can you be more specific!"

    similarity_values = cosine_similarity(user_tfidf,que_tfidf) #find similariy of user query with the corpus
    similarity_values=similarity_values[0].tolist()
    #print(similarity_values)
    
    q,r=0,0
    ilist=[]
    if(not sum(similarity_values)): #no similarity with questions
        similarity_values = cosine_similarity(user_tfidf,res_tfidf) #find similariy of user query with the corpus
        similarity_values=similarity_values[0].tolist()
        #print(similarity_values)
        indices=[i for i,x in enumerate(similarity_values) if x==max(similarity_values)]
        r=1

    else: #find similarity with questions
        indices=[i for i,x in enumerate(similarity_values) if x==max(similarity_values)]
        q=1
    
    #print(indices)
    resp_list=[]
    if len(indices)==1:
        if(similarity_values[indices[0]]==0):
            return "Sorry! I didn't understand you!"
        else:
            return responses[indices[0]]
    else:
        for i in range(len(indices)):
            if similarity_values[indices[i]]!=0:
                resp_list.append(responses[indices[i]])
            else:
                return 'No intents! Can you be more specific please!'
            
    return resp_list


# # Find response using entities


def get_entity_document_list(user_entities): #user_entities is a dictionary
    matching_document_indexes=[]
    
    if(len(user_entities)==0):
        return matching_document_indexes
    
    else:
        user_entity_list=[(key,value) for key,value in user_entities.items()] #make it a list of tuples
        #print(user_entity_list)
        
        for entity in user_entity_list: #check if there are any matching entities
            q,r = -1,-1
            try:
                q=que_entities.index(entity[0]) #check in question entities
                #print(q,que_entities[q])
            except:
                try:
                    r=res_entities.index(entity[0]) #check in response entities
                    #print(r,res_entities[r])
                except:
                    print("----No matching entities")
                    #return matching_document_indexes
            
            if q!=-1: #there is some match with the questions
                l=que_entities_index_list[q] #get the indexes of questions with matching entities
                matching_document_indexes.append(l)
                #print(l)

            elif q==-1 and r!=-1: #no match with questions, but some match with answers
                l=res_entities_index_list[r] #get the indexes of responses with matching entities
                l=[i+int(len(documents)/2) for i in l]
                matching_document_indexes.append(l)
                #print(l)
        
        final_index=[]
        for i in matching_document_indexes:
            final_index.extend(i)
        final_index=list(set(final_index))
        final_index.sort()
        #print(final_index)
        
        return final_index


def get_intent_document_list(entity_match_list,user_intents):  
    #print(entity_match_list)
    min_intents=2
    max_intents=5
    match_threshold=0.5
    
    intent_match_list=[]
    perfect_match_list=[]
    exact_match_list=[]
    if not entity_match_list: #no entities are matched --> return empty list
        return intent_match_list
    
    if len(user_intents)<=min_intents: #100% match with user intents
        threshold=len(user_intents)
    else:
        threshold= len(user_intents) * match_threshold #percentage of match (say 50% here)
    if threshold>max_intents:
        threshold=max_intents #if more than 5 intents, atleast 5 should match
    #print(threshold)
    
    for i in entity_match_list:
        if i<int(len(documents)/2):
            count=0
            for intent in user_intents: #intent matching
                if intent in que_intents[i]:
                    count+=1    
            #print(i,count)
            if count==len(user_intents) and set(que_intents[i])==set(user_intents):
                exact_match_list.append(i)
            if count==len(user_intents): #all intents matched --> perfect match
                perfect_match_list.append(i)
            elif count>=threshold: #sufficient intents matched --> intent match list
                intent_match_list.append(i)
            
    if not intent_match_list: #no matching in questions
        #print('not entered')
        for i in entity_match_list:
            count=0 
            if i>int(len(documents)/2)-1:
                i=i-int(len(documents)/2)
                for intent in user_intents: #intent matching
                    if intent in res_intents[i]:
                        count+=1    
                #print(i,count)
                if count==len(user_intents) and set(res_intents[i])==set(user_intents):
                    exact_match_list.append(i)
                if count==len(user_intents): #all intents matched --> perfect match
                    perfect_match_list.append(i)
                elif count>=threshold: #sufficient intents matched --> intent match list
                    intent_match_list.append(i)
            
    if exact_match_list:
        return exact_match_list
    elif perfect_match_list:
        return perfect_match_list
    elif intent_match_list:
        return intent_match_list
    else:
        return intent_match_list


def find_tfidf_for_intent_matched(intent_match_list,user_tfidf):
    docs=[]
    for i in intent_match_list:
        docs.append(documents[i]) #create a list of matching questions
    #print(docs)

    sim=[]
    for i in intent_match_list:
        if i<int(len(documents)/2):
            similarity_values = cosine_similarity(user_tfidf,que_tfidf[i]) #similarity computation
        else:
            similarity_values = cosine_similarity(user_tfidf,res_tfidf[i-int(len(documents)/2)]) #similarity computation
            
        similarity_values=similarity_values[0].tolist()
        #print(similarity_values)
        sim.append(similarity_values[0]) #list of similarity values
        
    #index=intent_match_list[sim.index(max(sim))] #find the matching document index from the dictionary list
    indices=[intent_match_list[i] for i,x in enumerate(sim) if x==max(sim)]
    
    resp_list=[]
    for i in indices:
        if(i<int(len(documents)/2)):
            resp_list.append(responses[i])
        else:
            resp_list.append(responses[i-int(len(documents)/2)])
            
    return resp_list
    #return index


def find_response_using_entities(user_intents,user_entities):        
    try:
        user_tfidf=find_user_tfidf(find_user_tf(user_intents))
        #print(user_tfidf)
    except:
        return "Sorry! I didn't understand you! No sufficient information!"
    
    entity_match_list=get_entity_document_list(user_entities) #find entity matching list
    
    if entity_match_list:
        #print(entity_match_list)
        intent_match_list=get_intent_document_list(entity_match_list,user_intents) #find intent matching list
        
        if intent_match_list: #find tfidf for only intent match list
            #matching_document_index=find_tfidf_for_intent_matched(intent_match_list,user_tfidf)
            return find_tfidf_for_intent_matched(intent_match_list,user_tfidf)
        
        else: # no intent matching list -->find response using normal tfidf
            print('I do not have sufficient information! But I can provide you with some related information')
            return find_response(user_intents)
    else:
        return find_response(user_intents)


from csv import reader #2_

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

    with open(corpus_file, 'r', encoding = "utf8") as read_obj:
        csv_reader = list(reader(read_obj))
        for row in csv_reader:
            if(row[0][-1].isalnum() == False):
                doc = nlp(row[0][:-1])
            else:
                doc = nlp(row[0])
            arr.append({})
            for ent in doc.ents:
                arr[i][str(ent)] = ent.label_

            if(row[1][-1].isalnum() == False):
                doc = nlp(row[1][:-1])
            else:
                doc = nlp(row[1])
            arr1.append({})
            for ent in doc.ents:
                arr1[i][str(ent)] = ent.label_
            i = i + 1
    
        read_obj.close()

    arr.remove(arr[0])
    arr1.remove(arr1[0])

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

    create_pickle('entities_and_labels_questions_dictionary_t.pickle', dis_que)
    create_pickle('entities_and_labels_responses_dictionary_t.pickle', dis_res)
    

def generate_entities_mapped_to_corpus_dictionary2(corpus_file): #4_new
    #load spacy model
    nlp = spacy.load("en_core_web_sm")
    ent_set = set()
    
    with open(corpus_file, 'r', encoding = "utf8") as read_obj:
        csv_reader = list(reader(read_obj))
        for row in csv_reader:
            if(row[0][-1].isalnum() == False):
                doc = nlp(row[0][:-1])
            else:
                doc = nlp(row[0])
            for ent in doc.ents:
                ent_set.add(str(ent).lower())

            if(row[1][-1].isalnum() == False):
                doc = nlp(row[1][:-1])
            else:
                doc = nlp(row[1])
            for ent in doc.ents:
                ent_set.add(str(ent).lower())
        read_obj.close()
    
    #print(ent_set)

    dis_ent_dic_que = {}
    dis_ent_dic_res = {}

    abb_questionlist = []
    abb_responselist = []
    with open("phrase_corrected_corpus_t.csv", 'r', encoding = "utf8") as read_obj:
        csv_reader = list(reader(read_obj))
        for row in csv_reader:
            abb_questionlist.append(row[0])
            abb_responselist.append(row[1])
    abb_questionlist.remove(abb_questionlist[0])
    abb_responselist.remove(abb_responselist[0])
    
    for i in range(0, len(abb_questionlist)):
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        words_list = tokenizer.tokenize(abb_questionlist[i])
        
        for word in words_list:
            j = word.lower()
            if j in ent_set:
                if j not in dis_ent_dic_que:
                    dis_ent_dic_que[j] = {i}
                else:
                    dis_ent_dic_que[j].add(i)
        
        words_list = tokenizer.tokenize(abb_responselist[i])
        for word in words_list:
            j = word.lower()
            if j in ent_set:
                if j not in dis_ent_dic_res:
                    dis_ent_dic_res[j] = {i}
                else:
                    dis_ent_dic_res[j].add(i)
                    
                
    create_pickle('entities_mapped_to_questions_dictionary_t.pickle', dis_ent_dic_que)
    create_pickle('entities_mapped_to_responses_dictionary_t.pickle', dis_ent_dic_res)

GREETING_INPUTS = ["hello", "hi", "greetings", "sup", "what's up","hey"]
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
  
def get_popular_ques(que_intents): #find all questions with no intents
    popular=[]
    for i in que_intents:
        if len(i)==0:
            popular.append(que_intents.index(i))
    return popular
    

def get_popular_ans(user_input): #compare user input with popular questions
    user_input=re.sub(r'\?','',user_input.lower())
    d=nltk.word_tokenize(user_input)
    
    a=get_popular_ques(que_intents)
    b=pd.read_pickle('questions2_t.pickle')
    resp_list=[]
    for i in a:
        b[i]=re.sub(r'\?','',b[i].lower())
        c=nltk.word_tokenize(b[i])
    
        if set(c)==set(d):
            resp_list.append(responses[i])
    return resp_list


# # Main function

def create_files(corpus):  # create required files
    print('--------Building Chatbot--------')
    print('Please wait until all the required files are generated!')
    
    if os.path.exists('questions_t.pickle')==False or os.path.exists('responses_t.pickle')==False:
        print('creating question files and response files..')
        extract_corpus_file(corpus) #generates questions_t.pickle and responses_t.pickle
        print('----question files and response files created..')

    if os.path.exists('abb_corpus_t.csv')==False:
        print('creating abbreviation files..')
        a=Abbreviations_Detection()
        a.Abbreviations(corpus) #creates supporting dictionary for abb check and also abbreviated corpus
        abb_check=pd.read_pickle('abbreviations_t.pickle')
        create_abbreviations_dict(abb_check) #creates a detected_abbreviations_t.pickle
        print('----abbreviation files created..')

    if os.path.exists('entities_and_labels_questions_dictionary_t.pickle')==False or os.path.exists('entities_and_labels_responses_dictionary_t.pickle')==False:
        print('creating entity files..')
        generate_entities_and_labels_dictionary('abb_corpus_t.csv') #2_

        entities=pd.read_pickle('entities_and_labels_questions_dictionary_t.pickle')
        en=pd.read_pickle('entities_and_labels_responses_dictionary_t.pickle')

        for i in en.keys():
            if i not in entities.keys():
                entities[i]=en[i]
        print('----entity files created..')

    if os.path.exists('phrase_corrected_corpus_t.csv')==False:
        print('creating phrase files..')  
        P=Phrase_Collection()
        sp,series=P.Phrases_with_hyphen('abb_corpus_t.csv')
        P.Phrase_collection('abb_corpus_t.csv',entities,sp,scoring='npmi')
      
        generate_phrases_list() #creates a list of phrases - detected_phrases_t.pickle

        phrases_list=pd.read_pickle('detected_phrases_t.pickle')
        generate_phrase_to_word_dict(phrases_list) #creates a dict of phrases and words - phrase_to_word_dictionary_t.pickle

        create_distinct_abb_tokens(pd.read_pickle('detected_phrases_t.pickle')) #creates a list of distinct abb tokens   

        generate_entities_mapped_to_corpus_dictionary2('phrase_corrected_corpus_t.csv') #4_
        print('----phrase files created..') 

    if os.path.exists('que_intents_t.pickle')==False or os.path.exists('ans_intents_t.pickle')==False:
        print('creating intent files..')
        get_intents()
        print('----intent files created..')

    if os.path.exists('syn_list_t.pickle')==False or os.path.exists('syn_ind_t.pickle')==False:
        print('creating synonym files..')
        get_words_synonyms()
        print('----synonym files created..')

    if os.path.exists('distinct_words_t.csv')==False:
        print('creating distinct words file..')
        find_distinct_words()
        print('----distinct words file created..')
    
    if os.path.exists('question_tfidf_vectors_t.pickle')==False or os.path.exists('response_tfidf_vectors_t.pickle')==False:
        print('creating tfidf vectors file..')
        generate_tfidf_vectors('que_intents_t.pickle','ans_intents_t.pickle','distinct_words_t.csv') #creates que_tfidf, res_tfidf, idf vectors
        print('----tfidf vectors file created..')

    if os.path.exists('single_word_definations_t_t.pickle')==False:
        print('creating one word descriptions file..')
        generate_single_word_definitions() #creates a dictionary of single words queries
        print('----one word descriptions file created..')

    if os.path.exists('corpus_t.txt')==False:
        print('building corpus dictionary..')
        generate_corpus_enc() #creates corpus_enc text file
        print('----corpus dictionary built..')

    print('\n*********** all files found ***********')

    
#merge duplicate questions for testing
def merge_duplicate_docs():
    que_documents=pd.read_pickle('questions_t.pickle')
    res_documents=pd.read_pickle('responses_t.pickle')

    q=[] #stores distinct questions
    for i in que_documents:
        if i not in q:
            q.append(i)

    values=np.array(que_documents)
    qindex=[]
    for i in range(len(q)):
        search=q[i]
        k=np.where(values==search)[0]
        qindex.append(k)

    r=[] #stores distinct responses list
    for i in qindex:
        rs=[]
        for j in i:
            rs.append(res_documents[j])
        r.append(rs)

    que_res={} #dictionary of distinct questions and responses
    for i in range(len(q)):
        que_res[q[i]]=r[i]
    #print(que_res)

    que_list=[] #list of 1301 questions
    ans_list=[] #list of 1301 answers
    for i in range(len(que_documents)):
        que_list.append(que_documents[i])
        ans_list.append(que_res[que_documents[i]])

    #len(que_list),len(ans_list)
    
    return que_list,ans_list,q,r


def test_chatbot(): #testing on all database questions
    correct_count = 0
    wrong_count = 0
    wrong_indices=[]
    spell_time=0
    abbrev_time=0
    phrase_time=0
    entity_time=0
    
    que_list,ans_list,q,r=merge_duplicate_docs()
    
    print('\n\n')
    print('\n\n--------TESTING ON ALL THE DATASET QUESTIONS--------\n')
    print('total no. of questions :', len(que_list))
    print('\n----------------------------------------------------------\n')
    
    for i in range(len(que_list)): #use que_list and ans_list for questions and responses
        user_input=que_list[i]
        print(i,'QUE:',user_input)
        
        if(user_input!='bye'):
            P=Phrase_Collection()
            user_input=P.replace_phrases(phrase_check2,abbreviation_check(abb_check,spell_check(user_input)))

            user_entities=get_entities2(user_input) #extract entities
            entities=[i.lower() for i in user_entities]
            user_intents=find_intents(user_input) #find user intents
            #print(user_intents)
            
            print("ROBO: ",end="")
              
            if len(user_intents)==0:
                resp=get_popular_ans(user_input)
                if not resp:
                    resp='No sufficient data!'
                    
            elif('fullform' in user_intents): #check if the query is of the form 'what is the fullform of XXXX'
                resp=find_fullform(user_intents)

            elif(len(user_entities)==1 and len(user_intents)==1 and entities[0].lower() in query): #single word queries
                if (user_intents[0] not in que_def):
                    resp=definition[query.index(entities[0].lower())]
                else:
                    resp=responses[que_intents.index([user_intents[0]])]
                
            elif len(user_entities)==0: #no entities in the user input --> so normal tfidf computation
                resp=find_response(user_intents)

            else: #use entities to find response
                resp=find_response_using_entities(user_intents,user_entities)
            
            print(resp)
            
            if type(resp)==str:
                if resp in ans_list[i]:
                    correct_count+=1
                else:
                    print('========================================================')
                    wrong_count+=1
                    wrong_indices.append(i)
                    print(ans_list[i])
            else:
                if (resp in ans_list[i]) or (ans_list[i] in resp) or (set(ans_list[i]) & set(resp)):
                    correct_count+=1
                else:
                    print('========================================================')
                    wrong_count+=1
                    wrong_indices.append(i)
                    print(ans_list[i])

        print('----------------------------------------------------------')
    print('correct:',correct_count)
    print('wrong:',wrong_count)
    print('wrong_answered:',end='')
    print(wrong_indices)

    
def test_chatbot_twisted(): #testing on twisted questions
    correct_count = 0
    wrong_count = 0
    wrong_indices=[]
    spell_time=0
    abbrev_time=0
    phrase_time=0
    entity_time=0
        
    test_data=pd.read_csv(test_data_file)
    test_questions=list(test_data['questions'])
    test_responses=list(test_data['responses'])
    
    print('\n\n--------TESTING ON TWISTED QUESTIONS--------\n')
    print('total no. of questions :', len(test_questions))
    print('\n----------------------------------------------------------\n')
    
    for i in range(len(test_questions)): #use q and r for questions and responses
        user_input=test_questions[i]
        print(i,'QUE:',user_input)
        
        if(user_input!='bye'):
            P=Phrase_Collection()
            user_input=P.replace_phrases(phrase_check2,abbreviation_check(abb_check,spell_check(user_input)))

            user_entities=get_entities2(user_input) #extract entities
            entities=[i.lower() for i in user_entities]
            user_intents=find_intents(user_input) #find user intents
            #print(user_intents)
            
            #print("ROBO: ",end="")
            if('fullform' in user_intents): #check if the query is of the form 'what is the fullform of XXXX'
                resp=find_fullform(user_intents)

            elif(len(user_entities)==1 and len(user_intents)==1 and entities[0].lower() in query): #single word queries
                if (user_intents[0] not in que_def):
                    resp=definition[query.index(entities[0].lower())]
                else:
                    resp=responses[que_intents.index([user_intents[0]])]

            elif len(user_entities)==0: #no entities in the user input --> so normal tfidf computation
                resp=find_response(user_intents)

            else: #use entities to find response
                resp=find_response_using_entities(user_intents,user_entities)

            print(resp)
                
            if type(resp)==str:
                if resp == test_responses[i]:
                    correct_count+=1
                else:
                    print('========================================================')
                    wrong_count+=1
                    wrong_indices.append(i)
                    print(test_responses[i])
            else:
                if (test_responses[i] in resp):
                    correct_count+=1
                else:
                    print('========================================================')
                    wrong_count+=1
                    wrong_indices.append(i)
                    print(test_responses[i])

            print('----------------------------------------------------------')
    print('correct:',correct_count)
    print('wrong:',wrong_count)
    print('wrong_answered:',end='')
    print(wrong_indices)
    

def run_chatbot(usertext): #function that starts chatbot
    flag=True
    print('\n\n ------- CHATBOT STARTED --------\n')
    print("BOT : Hey Hi! I am here to answer your queries. For acronyms, type 'fullform'. Type 'bye' to end conversation")
    
    # while(flag==True):
    user_input=usertext
    start_time=datetime.now() #timer starts
    
    if(user_input!='bye'):
        P=Phrase_Collection()
        user_input=P.replace_phrases(phrase_check2,abbreviation_check(abb_check,spell_check(user_input)))
        
        user_entities=get_entities2(user_input) #extract entities
        entities=[i.lower() for i in user_entities]
        #print(user_entities)
        
        if(user_input in ['thanks','thank you','thankyou'] ):
            flag=False
            print("ROBOT: You are welcome...")
            return "ROBOT: You are welcome..."
        else:
            if(greeting(user_input)!=None): #greeting
                print("ROBOT: "+greeting(user_input))
                return "ROBOT: "+greeting(user_input)
            else:
                user_intents=find_intents(user_input) #find user intents
                #user_intents=change_phrase_to_word(user_intents)
                
                for i in range(len(user_intents)):
                    if user_intents[i] not in distinct_words:
                        try:
                            user_intents[i]=distinct_words[syn_index[synonyms.index(user_intents[i])]]  
                        except:
                            continue
                #print(user_intents)
                
                # print("ROBOT: ",end="")
                # return "ROBOT: "

                if len(user_intents)==0: #check popular questions
                    resp=get_popular_ans(user_input)
                    if not resp:
                        resp='No sufficient data!'
                
                elif('fullform' in user_intents): #check if the query is of the form 'what is the fullform of XXXX'
                    try:
                        resp=find_fullform(user_intents)
                    except ValueError: #abbreviation not found in the corpus
                        resp='Sorry! I do not have the requested information!'

                elif(len(user_entities)==1 and len(user_intents)==1 and entities[0].lower() in query): #single word queries
                    if (user_intents[0] not in que_def):
                        resp=definition[query.index(entities[0].lower())]
                    else:
                        resp=responses[que_intents.index([user_intents[0]])]

                elif len(user_entities)==0: #no entities in the user input --> so normal tfidf computation
                    resp=find_response(user_intents)

                else: #use entities to find response
                    resp=find_response_using_entities(user_intents,user_entities)
                    
                if type(resp)==str: #only one response
                    print(resp) 
                    return "ROBOT: "+ resp
                else: #pick a random response from a list of responses
                    print(list(set(resp)))
                    return "ROBOT: "+ list(set(resp))
        print('---------------------------')
        end_time=datetime.now() #timer starts
        print('time taken(seconds):',(end_time-start_time).total_seconds())
        # return 'time taken(seconds):',(end_time-start_time).total_seconds()
        print('----------------------------------------------------------')
        # return '----------------------------------------------------------'
        
    else: #end conversation
        flag=False
        print("ROBOT: Bye! take care...")
        return "ROBOT: Bye! take care..."

    
# ========create required files for the chatbot========
def tfidf_run():

    global corpus_file
    corpus_file='QuestionsResponsesISRO.csv' #file name of original corpus
    start_time=datetime.now()
    create_files(corpus_file)
    end_time=datetime.now()
    print('time(minutes):',(end_time-start_time).total_seconds()/60)
    print('\n--------READY TO TEST THE CHATBOT----------')


    # ========fetch the data from created files========
    global questions,responses,abb_corpus,phrase_corrected_corpus,documents,distinct_words
    global que_intents,res_intents,que_def,res_def,idf,que_tfidf,res_tfidf
    questions=pd.read_pickle('questions_t.pickle')
    responses=pd.read_pickle('responses_t.pickle')

    abb_corpus=pd.read_csv('abb_corpus_t.csv')
    phrase_corrected_corpus=pd.read_csv('phrase_corrected_corpus_t.csv')

    documents=pd.read_pickle('doc_t.pickle')

    distinct_words=list(pd.read_csv('distinct_words_t.csv')['words'])
    que_intents=pd.read_pickle('que_intents_t.pickle')
    res_intents=pd.read_pickle('ans_intents_t.pickle')

    que_def=pd.read_pickle('que_def_t.pickle')
    res_def=pd.read_pickle('res_def_t.pickle')

    idf=pd.read_pickle('idf_vector_t.pickle')
    que_tfidf=pd.read_pickle('question_tfidf_vectors_t.pickle')
    res_tfidf=pd.read_pickle('response_tfidf_vectors_t.pickle')

    global synonyms,syn_index,syn_list,e1,que_entities,e2,res_entities,que_entities_index_list,res_entities_index_list

    synonyms=list(pd.read_pickle('syn_ind_t.pickle').keys())
    syn_index=list(pd.read_pickle('syn_ind_t.pickle').values())

    distinct_words=list(pd.read_pickle('syn_list_t.pickle').keys())
    syn_list=list(pd.read_pickle('syn_list_t.pickle').values())

    e1=pd.read_pickle('entities_mapped_to_questions_dictionary_t.pickle')
    que_entities=[ent for ent in e1]
    que_entities_index_list=[list(e1[ent]) for ent in e1]

    e2=pd.read_pickle('entities_mapped_to_responses_dictionary_t.pickle')
    res_entities=[ent for ent in e2]
    res_entities_index_list=[list(e2[ent]) for ent in e2]

    global arr,arr1

    arr = {} #entity extraction
    arr1 = {}
    with open('entities_mapped_to_questions_dictionary_t.pickle', 'rb') as read_obj:
        pkl_reader = pickle.load(read_obj)
        for i in pkl_reader:
            arr[i] = pkl_reader[i]
        read_obj.close()
    with open('entities_mapped_to_responses_dictionary_t.pickle', 'rb') as read_obj:
        pkl_reader = pickle.load(read_obj)
        for i in pkl_reader:
            arr1[i] = pkl_reader[i]
        read_obj.close()

    global corpus_dict,standard_dict,dic,abb_check,detected_abbreviations,fullform,shortform,distinct_abb
    global phrase_check2,phrases_list,phrases_dict,series,query_definition,query,definition,test_data_file
    corpus_dict = enchant.PyPWL("corpus_t.txt")
    standard_dict = enchant.Dict("en_US")
    dic = pd.read_pickle("words_set_t.pickle")

    abb_check=pd.read_pickle('abbreviations_t.pickle')
    detected_abbreviations=pd.read_pickle('detected_abbreviations_t.pickle')
    fullform=list(detected_abbreviations.keys())
    shortform=list(detected_abbreviations.values())

    distinct_abb=create_distinct_abb_tokens(list(detected_abbreviations.keys()))
    distinct_abb=pd.read_pickle('distinct_abb_t.pickle')

    phrase_check2=pd.read_pickle('phrase_check_t.pickle')
    phrases_list=pd.read_pickle('detected_phrases_t.pickle')
    phrases_dict=pd.read_pickle('phrase_to_word_dictionary_t.pickle')

    series=pd.read_pickle('series_t.pickle')

    query_definition=pd.read_pickle('single_word_definations_t_t.pickle')
    query=[i.lower() for i in query_definition.keys()]
    definition=[i for i in query_definition.values()]

    test_data_file='test_questions.csv' #file name of twisted questions

    # ========testing on database questions========
    start_time=datetime.now()
    test_chatbot() 
    end_time=datetime.now()
    print('----------------------------------------------------------')
    print('time taken(seconds):',(end_time-start_time).total_seconds())


    # ========testing on twisted questions========
    s=datetime.now()
    test_chatbot_twisted() 
    e=datetime.now()
    print('time taken(seconds):',(e-s).total_seconds())


    # ========start chatbot========
    # print(run_chatbot())

    # while(True):
    #     print(run_chatbot())
    #     end1=int(input("enter 1 to end "))
    #     if(end1==1):
    #         break



# tfidf_run()



