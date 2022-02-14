#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[2]:


import nltk
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import RegexpTokenizer
from gensim.models import Word2Vec
import math
import pandas as pd
from num2words import num2words
from itertools import chain
import math
import enchant #used in spellchecking
import sys
# import spellchecker
from spellchecker import SpellChecker
import re
import random
import spacy
from spacy import displacy
from timeit import default_timer as timer
import os
from csv import reader
import pickle
import codecs
from word2number import w2n


# # getting the list of que and ans

# In[3]:


# csv_file="QuestionsResponsesISRO.csv"

#method for generating pickle file
def pickle_create(name,data): 
    with open(name, 'wb') as handle:
        pickle.dump(data,handle, protocol=2)
#method for geeting list of questions and answers
def get_que_ans(csv_file,questions,answers,):
    dataset=pd.read_csv(csv_file)
    que_list=[]                      #list  questions
    ans_list=[]                       #list  responses
    que_list.extend(list(dataset[questions]))
    ans_list.extend(list(dataset[answers]))
    pickle_create("que_list_w.pickle",que_list)
    pickle_create("ans_list_w.pickle",ans_list)
    print("created file : \t que_list_w.pickle ,ans_list_w.pickle")

#get_que_ans(csv_file,"questions","responses",path_ans)
    
def get_que_ans_test(csv_file,questions,answers):
    dataset=pd.read_csv(csv_file)
    que_list=[]                      #list  questions
    ans_list=[]                       #list  responses
    que_list.extend(list(dataset[questions]))
    ans_list.extend(list(dataset[answers]))
    pickle_create("que_list_test_w.pickle",que_list)
    pickle_create("ans_list_test_w.pickle",ans_list)
    print("created file : \t que_list_test_w.pickle ,ans_list_test_w.pickle")
    
def get_que_ans_db(csv_file,questions,answers):
    dataset=pd.read_csv(csv_file)
    que_list=[]                      #list  questions
    ans_list=[]                       #list  responses
    que_list.extend(list(dataset[questions]))
    ans_list.extend(list(dataset[answers]))
    pickle_create("que_list_db_w.pickle",que_list)
    pickle_create("ans_list_db_w.pickle",ans_list)
    print("created file : \t que_list_db_w.pickle ,ans_list_db_w.pickle")


# In[ ]:





# # Intents and distinct words generation

# In[7]:


def tokenize_words_pos(text):
    tokenized_words=[]
    stopwrds=stopwords.words('english')
    #stopwrds.remove('no')                  #include are exclude words according to the domain
    stopwrds.remove('not')
    #stopwrds.append('can')
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer() 
    for i in text:
        wrd_token=nltk.pos_tag(tokenizer.tokenize(i))      #converting to lower case
        wrd_token=[(lemmatizer.lemmatize(lemmatizer.lemmatize(word[0]),pos='v'),word[1]) for word in wrd_token if word[0] not in stopwrds] #stopword removal and lemmatization
        wrd_token=[(word[0].lower(),word[1]) for word in wrd_token ]  
        if not wrd_token:
            wrd_token=nltk.pos_tag(tokenizer.tokenize(i))  #if the intent is empty tokenizing the sentence only
        
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
    
    que_list=pd.read_pickle("que_list_w.pickle")
    ans_list=pd.read_pickle("ans_list_w.pickle")
    
    que_intents_pos=tokenize_words_pos(que_list)
    ans_intents_pos=tokenize_words_pos(ans_list)
    
    corpus_intents_pos=que_intents_pos+ans_intents_pos
    lt=['NNP']
    nouns_list=[]
    for i in corpus_intents_pos:
            for k in i:
                if k[1] in lt and k[0].lower() not in stopwords.words('english'):
                    if k[0] not in nouns_list:
                        nouns_list.append(k[0])
    
    pickle_create("nouns_list_w.pickle",nouns_list)
    
    print("created file : \t nouns_list_w.pickle")
    
def get_intents():
    
    que_list=pd.read_pickle("que_list_w.pickle")
    ans_list=pd.read_pickle("ans_list_w.pickle")
    
    que_intents_pos=tokenize_words_pos(que_list)
    ans_intents_pos=tokenize_words_pos(ans_list)
    
    que_intents=derive_intents(que_intents_pos)
    ans_intents=derive_intents(ans_intents_pos)
    
    pickle_create("que_intents_w.pickle",que_intents)
    pickle_create("ans_intents_w.pickle",ans_intents)
    
    pickle_create("que_intents_pos_w.pickle",que_intents_pos)
    pickle_create("ans_intents_pos_w.pickle",ans_intents_pos)
    
    que_distinct_words=make_list(que_intents)
    ans_distinct_words=make_list(ans_intents)
    
    corpus_distinct_words=que_distinct_words+ans_distinct_words
    
    pickle_create("que_distinct_words_w.pickle",que_distinct_words)
    pickle_create("ans_distinct_words_w.pickle",ans_distinct_words)
    pickle_create("corpus_distinct_words_w.pickle",corpus_distinct_words)
    
    
    print("created files \t :","que_intents_w.pickle","ans_intents_w.pickle","que_intents_pos_w.pickle","ans_intents_pos_w.pickle","que_distinct_words_w.pickle","ans_distinct_words_w.pickle","corpus_distinct_words_w.pickle")
    
    


# # Synonyms Generation

# In[8]:


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
        if (getPathSimilarity(wrd,m)>=0.5):
            #print(j,m)
            if m not in fl:
                fl.append(m)
    if len(fl)==0:
        fl.append(wrd)
        
    return fl

def corpus_dict_pos(que_tokens,que_distinct_words):
    nd={}
    nd1={}
    nb=len(que_tokens)
    mk=[]
    fl=[]
    for i in range(0,nb): 
        #print(i)
        for k1 in que_tokens[i]:
            fl=[]
            lt1=[]
            j=k1[0]  #getting the word
            pos_tag=k1[1][0].lower()    #getting the pos tag of word
            if k1[0]  in mk:
                #print("word",k1[0])
                mk.append(k1[0])           #finding the synonym of the word if not seen before
                lt1.append(nd[j])
            if pos_tag=='j':
                pos_tag='a'
            if pos_tag  in ['n','a','r','v','s']:  #finding the synonyms with the following pos tags due wordnet constraint
                lt1=lt1+get_synonyms(j,pos_tag)      #getting the synonyms
                fl=[]
                for m in lt1:       #if the word is not a corpus word condition to avoid conflict
                        if m not in fl:
                            fl.append(m)
                if j not in fl:
                    fl.append(j)    #if the filtration leads to empty list appending the list with the corpus word itself
                nd[j]=fl 

            elif pos_tag=='c' and j.isdigit():
                fl=[]     #finding the full word of the numbers
                fl.append(num2words(j))
                if len(fl)==0:
                    fl.append(j)
                nd[j]=fl
            else:
                nd[j]=[j]

            for k in nd[j]:
                lt2=[]
                if k in nd1.keys():
                    #print(j,i)
                    lt2=nd1[k]
                    if j!=k:
                        ps=getPathSimilarity(j,k)
                    else:
                        ps=1
                    if (j,i,ps) not in lt2:
                        nd1[k].append((j,i,ps))
                else:
                    #print("here",j,i)
                    if j!=k:
                        ps=getPathSimilarity(j,k)
                    else:
                        ps=1
                    lt2.append((j,i,ps))
                    nd1[k]=lt2
    return nd,nd1

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
    
    st= timer()
    
    que_intents_pos=pd.read_pickle("que_intents_pos_w.pickle")
    ans_intents_pos=pd.read_pickle("ans_intents_pos_w.pickle")
    
    que_distinct_words=pd.read_pickle("que_distinct_words_w.pickle")
    ans_distinct_words=pd.read_pickle("ans_distinct_words_w.pickle")
    
    que_syn_dict,que_syn_dict_index=corpus_dict_pos(que_intents_pos,que_distinct_words)
    #ans_syn_dict,ans_syn_dict_index=corpus_dict_pos(ans_intents_pos,ans_distinct_words)
    
    pickle_create("que_syn_dict_w.pickle",que_syn_dict)
    #pickle_create("ans_syn_dict_w.pickle",ans_syn_dict)
    
    pickle_create("que_syn_dict_index_w.pickle",que_syn_dict_index)
    #pickle_create("ans_syn_dict_index_w.pickle",ans_syn_dict_index)
    
    
    #syn_list= {**que_syn_dict, **ans_syn_dict}
    phrases_list=get_phrases_list(que_syn_dict)
    
    pickle_create("syn_list_w.pickle",que_syn_dict)
    pickle_create("phrases_list_w.pickle",phrases_list)
    
    
    ed =timer()
    print(ed-st)
    
    print("created files :\t","que_syn_dict_w.pickle","ans_syn_dict_w.pickle","syn_list_w.pickle","phrases_list_w.pickle")


# # TF IDF generation of words

# In[9]:


def tf_words(tks):
    nd={}
    for i in tks:
        for j in i:
            nd[j]=(i.count(j))/len(i)
    return nd

def idf_words(wrds,tks):
    nd1={}
    for i in wrds:
        c=0
        for j in tks:
            if i in j:
                c+=1
        nd1[i]=(math.log2(len(tks))/(1+c))
    return nd1

def tf_idf(nd,nd1):
    nd3={}
    tf=nd.keys()
    idf=nd1.keys()
    for i,j in zip(tf,idf):
        nd3[i]=nd[i]*nd1[j]
    return nd3

def tf_idf_words():

    que_intents=pd.read_pickle("que_intents_w.pickle")
    ans_intents=pd.read_pickle("ans_intents_w.pickle")
    
    que_distinct_words=pd.read_pickle("que_distinct_words_w.pickle")
    ans_distinct_words=pd.read_pickle("ans_distinct_words_w.pickle")
    
    
    que_tf=tf_words(que_intents)
    que_idf=idf_words(que_distinct_words,que_intents) 
    que_tfidf=tf_idf(que_tf,que_idf)
    
    ans_tf=tf_words(ans_intents)
    ans_idf=idf_words(ans_distinct_words,ans_intents)
    ans_tfidf=tf_idf(ans_tf,ans_idf)
    
    pickle_create("que_tfidf_w.pickle",que_tfidf)
    pickle_create("ans_tfidf_w.pickle",ans_tfidf)
    
    print("created files :\t ","que_tfidf_w.pickle","ans_tfidf_w.pickle")
    


# # Creating word2vec question vectors

# In[10]:


def tfidf_weighted_w2v(dt,ky,ml,tf):
    lb=[]
    cnt=[]
    vt=[]
    for i in dt:
        c=0
        vt=[]
        for j in i:
            for k in ky:
                if j==k: #checking if the word is in the dictionary
                    vt.append(tf[j]*ml.wv[k]) #finding the word vector and multiplying with the its tf idf score
                    c+=tf[j]         #saving the score of the words
                    break
        cnt.append(c)
        lb.append(vt) #word vector list of document vectors
    return lb,cnt

def make_sum(vq,vc):
    sq=[]
    for i in vq:
        s=0
        for j in i:
            s=s+j
        sq.append(s)       #adding the word vectors
    qt=[]
    for c,s in zip(vc,sq):
        qt.append(s/c)     #dividing word vectors with the no of words
    return qt

def shape_dim(x):
    bn=[]                         #shaping the vectors for further processing
    for i in range(0,len(x)):
        bn.append(np.array([x[i]]))
    return bn

def word2vec_modeling():
    
    
    que_intents=pd.read_pickle("que_intents_w.pickle")
    ans_intents=pd.read_pickle("ans_intents_w.pickle")
    
    que_distinct_words=pd.read_pickle("que_distinct_words_w.pickle")
    ans_distinct_words=pd.read_pickle("ans_distinct_words_w.pickle")
 
    que_tfidf=pd.read_pickle("que_tfidf_w.pickle")
    ans_tfidf=pd.read_pickle("ans_tfidf_w.pickle")

    
    model_que=Word2Vec(que_intents,min_count=1,window=3)
    model_ans=Word2Vec(ans_intents,min_count=1,window=3)
    
    saved_model_que_w= pickle.dumps(model_que) # saving the  pickled model
    pickle_create("saved_model_que_w",saved_model_que_w)
    saved_model_ans_w= pickle.dumps(model_ans) # saving the  pickled model
    pickle_create("saved_model_ans_w",saved_model_ans_w)
    
    que_vectors,que_wt_count=tfidf_weighted_w2v(que_intents,que_distinct_words,model_que,que_tfidf) 
    ans_vectors,ans_wt_count=tfidf_weighted_w2v(ans_intents,ans_distinct_words,model_ans,ans_tfidf) 
    
    que_vectors_sum=make_sum(que_vectors,que_wt_count)
    ans_vectors_sum=make_sum(ans_vectors,ans_wt_count)
    
    que_vectors_total=shape_dim(que_vectors_sum)
    ans_vectors_total=shape_dim(ans_vectors_sum)
    
    pickle_create("que_vectors_w.pickle",que_vectors_total)
    pickle_create("ans_vectors_w.pickle",ans_vectors_total)
    
    print("created files  :\t","saved_model_que_w","saved_model_ans_w","que_vectors_w.pickle","ans_vectors_w.pickle")


# # User query vectorization and Response finding

# In[11]:


def find_response_match_v1(user_query,l1,que_vector,que_list,ans_list):
    sim=[]
    for i in l1:
        val=cosine_similarity(user_query,que_vector[i])  #finding the cosine score of the user query and the doc vectors
        val=float(val[0])
        #print(val)
        sim.append(val)
    print(sim)
    ms=max(sim)    #max cosine score of the document
    ind=l1[sim.index(ms)]
    res=ans_list[ind]
    return res,ms

def find_response_match_v2(dt,que_vector,que_list,ans_list):
    sim=[]
    max_sim=[]
    ind1=[]
    ind=[]
    res=[]
    lt1=list(dt)
    for i,j in dt.items():
        val=cosine_similarity(que_vector[i],j)  #finding the cosine score of the user query and the doc vectors
        val=float(val[0])
        #print(val)
        sim.append(val)
    #print(sim)
    ms=max(sim) #max cosine score of the document
    max_sim=[i for i,x in enumerate(sim) if x==ms]
    #print(max_sim)
    for i in max_sim:
            ind.append(lt1[i])
    for i in ind:
        res.append(ans_list[i])
    return res,ind,ms 

def user_vectorize(uql,que_dict,tf,model_uq):
    #uql=token_wrd([user_query])
    uq=[]
    for i in uql:      #query tokenization
        for j in i:
            uq.append([j])
    #print(uq)
    uqv,uqc=tf_weighted_user_query(uq,que_dict,model_uq,tf)  #finding the vector of the user query
    uqt=[]
    s=0
    for i in range(len(uqv)): #adding the matrix
        s=s+uqv[i][0]
    uqt.append(s)
    uqtc=[]
    sc=0
    sc=sum(uqc)
    if sc>0:
        for s1 in uqt:
            uqtc.append(s1/sc)      #dividing the matrix with sum(tf idf) scores
        uq_vector=shape_dim_user(uqtc)   #shaping the matrix
        return uq_vector
    else:
        return np.zeros((1,100)) 
    
def tf_weighted_user_query(dt,ky,ml,tf):
    lb=[]
    cnt=[]
    vt=[]
    for i in dt:
        c=0
        vt=[]
        for k in ky.keys():
            lt=ky[k]
            if i[0] in ky:  #looking if the word is in synonyms list
                #print("in list ",i[0])
                vt.append(tf[i[0]]*ml.wv[i[0]])  #multiplying word with it tf idf score 
                c+=tf[k]
                break
            elif i[0] in lt:
                #print("in list ",i[0],k)
                vt.append(tf[k]*ml.wv[k])  #multiplying word with it tf idf score 
                c+=tf[k]
                break
        if len(vt)!=0 and c!=0:
            cnt.append(c)
            lb.append(vt)
    return lb,cnt

def shape_dim_user(x):
    bn1=[]
    for i in range(0,len(x)):
        bn1=(np.array([x[i]]))
    return bn1


# # Bot functions

# In[12]:


#greeting code for the chatbot
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
import random
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# # abbreviations 

# In[13]:


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
         
        with open('abb_corpus_w.csv','w',newline='',encoding='utf-8') as file:
            writer=csv.writer(file)
            writer.writerow(['questions','responses'])
            for i in range(len(questions)):
                writer.writerow([questions[i],responses[i]])
        
        
        filename1 = 'abbreviations_w.pickle' 
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
        
        with open('abbreviated_corpus_w.csv','w',newline='',encoding='utf-8') as file:
            writer=csv.writer(file)
            writer.writerow(['questions','responses'])
            for i in range(len(questions)):
                writer.writerow([questions[i],responses[i]])
        with open('abbreviations_w.csv','w',newline='',encoding='utf-8') as file:
            writer=csv.writer(file)
            writer.writerow(['full form','short form'])
            for i in abbreviations.keys():
                writer.writerow([i,abbreviations[i]])
                
        filename1 = 'abbreviations_w.pickle' 
        outfile = open(filename1,'wb') 
        pickle.dump(abbreviations1,outfile) 
        outfile.close()
        
        return 'abbreviated_corpus_w.csv',filename1,'abbreviations_w.csv'


# In[14]:


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

    pickle_create('entities_and_labels_questions_dictionary_w.pickle', dis_que)
    pickle_create('entities_and_labels_responses_dictionary_w.pickle', dis_res)


# In[15]:


def get_entities():
    
    en=pd.read_pickle('entities_and_labels_questions_dictionary_w.pickle')
    entities=pd.read_pickle('entities_and_labels_responses_dictionary_w.pickle')
    for i in en.keys():
        if i not in entities.keys():
            entities[i]=en[i]
    
    return entities


# In[16]:


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
            

        with open('phrase_corpus_w.csv','w',newline='',encoding='utf-8')as file:
            writer=csv.writer(file)
            writer.writerow(['questions','responses'])
            for i in range(len(questions)):
                writer.writerow([questions[i],responses[i]])
        
        filename1 = 'phrases_w.pickle' 
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
        
        filename1='series_w.pickle'
        outfile=open(filename1,'wb')
        pickle.dump(series,outfile)
        outfile.close()
        
        with open('phrase_w.csv','w',newline='',encoding='utf-8')as file:
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
            

        with open('phrase_corpus_w.csv','w',newline='',encoding='utf-8')as file:
            writer=csv.writer(file)
            writer.writerow(['questions','responses'])
            for i in range(len(questions)):
                writer.writerow([questions[i],responses[i]])
                
        with open('phrase_w.csv','a',newline='',encoding='utf-8')as file:
            writer=csv.writer(file)
            writer.writerow(['phrases'])
            for i in phrase:
                writer.writerow([i])

        filename1='phrases_w.pickle'
        outfile=open(filename1,'wb')
        pickle.dump(sp,outfile)
        outfile.close()

        return 'phrase_corpus_w.csv','phrases_w.pickle','phrase_w.csv'
    
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


# In[17]:


def get_series():
    series=set()
    with open('series_w.pickle', 'rb') as f: 
        series1 = pickle.load(f)

    with open('nouns_list_w.pickle', 'rb') as f: 
        nouns = pickle.load(f)

    for i in series1:
        series.add(i)
    for i in nouns:
        series.add(i)
        
    return series


# In[18]:


def get_definations_w(i):# i is list of nouns in caps or mixed
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
        
    definations_w={}
    for j in i:
        text=get_from_web(j)
        if text!='not available':
            text=text.replace('\n','')

            text=text.encode('ascii',errors='ignore').decode('ascii')
            sent_tokens=[]
            [sent_tokens.append(sent) for sent in nltk.sent_tokenize(text)]
            for k in range(2):
                try:
                    if j.lower() not in definations_w.keys():
                        definations_w[j.lower()]=sent_tokens[k]+' '
                    else:
                        definations_w[j.lower()]+=sent_tokens[k]+' '
                except:
                    pass
    filename='definations_w_w'
    output=open(filename,'wb')
    pickle.dump(definations_w,output)
    output.close()
        
        
    return definations_w


# # spell checker

# In[19]:


def build_corpus_dictionary(corpus_file):   
    if os.path.exists("corpus_w.txt"):
        os.remove("corpus_w.txt")

    #text file that stores the distinct words
    corpus_dict = open("corpus_w.txt", "a", encoding = 'cp1252')
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
    syn = pd.read_pickle('syn_list_w.pickle')
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
    abb = pd.read_pickle('abbreviations_w.pickle')
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
    di = pd.read_pickle('phrases_w.pickle')
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
    with open('single_words_dict_w.pickle', 'rb') as read_obj:
        pkl_reader = pickle.load(read_obj)
        for items in pkl_reader.items():
            tokenizer = nltk.RegexpTokenizer(r"\w+")
            words_list = tokenizer.tokenize(str(items))
            for word in words_list:
                if word.isnumeric() == False:
                    words_set.add(word)
                    words_set.add(word.lower())


    pickle_create("words_set_w.pickle", list(words_set))

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


# In[20]:


def create_pickle(file, data):
    open_file = open(file, 'wb')
    pickle.dump(data, open_file)
    open_file.close()
    
def spell_check(user_input,dic,standard_dict,corpus_dict): 
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


# In[21]:


def get_fullforms(abb_check):
    nd={}
    for i in abb_check.values():
        for k,j in zip(i.keys(),i.values()):
            nd[j]=k
    pickle_create('abbreviations_fullforms_dict_w.pickle', nd)

def find_fullform(tokens,fullform_list):
    for token in tokens:
        if(token!='full_form'):
            if token.upper() in fullform_list.keys():
                print("BOT : "+fullform_list[token.upper()])
                return 1
            else:
                print("BOT : "+"could not find abbreviation")
                return 0

def abbreviation_check(abbreviations,i):
        import re
        # function to call to replace abbreviation in user query
        y=i.split()
        for j in y:
            l={}
            for k in abbreviations.keys():
                if k.lower()==j or k==j:
                    d=abbreviations[k]
                    for m in d.keys():
                        if m.lower() in i:
                            i=i.replace(m.lower(),d[m])
                            break
                        elif m in i:
                            i=i.replace(m,d[m])
                            break
                    break
        i=re.sub(r'[()]',' ',i)
        x=i.split()
        for k in range(len(x)):
            try:
                if x[k]==x[k+1]:
                    x.remove(x[k+1])
            except:
                pass
            
        return ' '.join(w for w in x)
    
def single_word_def(user,single_wrd_dict):
    if user in single_wrd_dict.keys():
        print("BOT : "+single_wrd_dict[user])
        return 1
    else:
        print("BOT : could not find proper answers")
        return 0


# # Intents match

# In[22]:


def get_synonyms_intents(wrd):
    lt1=[]
    fl=[]
    [lt1.append(lem.name().lower()) for syn in wordnet.synsets(wrd) for lem in syn.lemmas()]
    [lt1.append(hypo.lower()) for hypo in getHyperNyms(wrd)]
    for m in lt1:
        if m not in fl:
            fl.append(m)
    if len(fl)==0:
        fl.append(wrd)
        
    return fl


# In[35]:


def getIntentsInetrsectingUserIntent(user_intent,que_syn_dict_index,que_syn_dict):
    intent_set={}
    nd={}
    nd1={}
    flag=0
    for word in user_intent[0]:
        #print("word",word)
        user_syn_tokens=[]
        if word not in que_syn_dict_index :
            #print("not in")
            user_syn_tokens=get_synonyms_intents(word)
            for syn in user_syn_tokens:
                if syn in que_syn_dict_index:
                    flag=1
                    for qn in que_syn_dict_index[syn]:
                        #print("qn",qn)
                        if syn==qn[0]:
                            #print("qn[1]",qn[1])
                            ps=1
                        else:
                        #print("word.lower(),qn[1]",word.lower(),qn[1])
                            ps=getPathSimilarity(word,qn[0])
                        #print("here",qn[0],qn[1])
                        if word not in intent_set:
                            #print("intents",word)
                            intent_set[word]={}
                            intent_set[word][qn[1]]=ps
                        elif qn[1] not in intent_set[word]:
                            #print("intents 2",word,qn[1])
                            intent_set[word][qn[1]]=ps
                        if word not in nd:
                            if ps>=0.5:
                                nd[word]=[(qn[0],qn[1],ps)]
                        else:
                            if ps>=0.5:
                                nd[word].append((qn[0],qn[1],ps))
        elif word in que_syn_dict_index:
            #print("in")
            for i in que_syn_dict_index[word]:
                if word not in intent_set:
                    #print("intents",word)
                    intent_set[word]={}
                    intent_set[word][i[1]]=i[2]
                elif i[1] not in intent_set[word]:
                    #print("intents 2",word,qn[1])
                    intent_set[word][i[1]]=i[2]

    #print('intents matching:',intent_set)
    return intent_set,flag,nd


# In[24]:


def getMatchingIntentsForUserIntent(intent_set,user_intent, match_threshold,min_lim,max_lim):
    #print(len(user_intent[0]))
    if not intent_set:
        return
    if len(user_intent[0])<=min_lim:
        min_match=len(user_intent)
    elif len(user_intent[0])>=max_lim:
        min_match=max_lim
    else:
        min_match=len(user_intent[0])*match_threshold
    threshold=min_match
    intent_score={}
    for item in intent_set:
        for qn in intent_set[item]:
            if qn in intent_score:
                intent_score[qn]=intent_score[qn]+intent_set[item][qn]
            else:
                intent_score[qn]=intent_set[item][qn]
    #print("after threshold:", intent_score)
    xv=get_intent_index(intent_score)
    #print("-------",xv)
    return xv


# In[25]:


def get_intent_index(ltx):
    ltx1={k: v for k, v in sorted(ltx.items(), key=lambda item: item[1])}
    res = dict(reversed(list(ltx1.items())))
    #print(res)
    xy=max(res.values())
    xy1=xy-0.5
    lty=[]
    for i in res:
        if res[i]==xy:
            lty.append(i)
        elif res[i]==xy1:
            lty.append(i)
    return lty


# In[26]:


def get_intents_index_final(user_intents,sdi,sd):
    xv,flag,nd=getIntentsInetrsectingUserIntent(user_intents,sdi,sd)
    if not bool(xv):
        xc=[]
        return xc,flag,nd
    xc=getMatchingIntentsForUserIntent(xv,user_intents,0.5,2,5)
    return xc,flag,nd


# In[27]:


def vectorized_dictionary(lt,user_query,user_intents,que_list,que_syn_dict,que_tfidf,model_que):
    nd={}
    for i in lt:
        flg=0
        iq=derive_intents(tokenize_words_pos([que_list[i]]))
        #print("que ",iq)
        #print("uq",user_intents[0])
        for j in user_intents[0]:
            #print(j)
            if j in que_syn_dict:
                #print("get syns")
                lt=que_syn_dict[j]
                #print(lt)
                for k in iq[0]:
                    if k in lt and k not in que_syn_dict:
                        #print("hee-----------------------------------------")
                        user_query_new=user_query.replace(j,k)
                        flg=1
                        break
        if flg==0:
            user_query_new=user_query
        #print(user_query_new)
        user_intents_new=derive_intents(tokenize_words_pos([user_query_new]))
        #print("vect query",user_intents)
        nd[i]=user_vectorize(user_intents_new,que_syn_dict,que_tfidf,model_que)
    return nd


# # Chatbot creation

# In[28]:


def create_chatbot(csv_file):
    
    st=timer()
    print("*********creating  files *******************")
    
    if os.path.isfile("que_list_db_w.pickle")==False :
        print("questions and response list creating.....")
        get_que_ans_db(csv_file,'questions','responses')
        print("questions and response list created")
    else:
        print("questions and response list exist")
    
    if os.path.isfile("abbreviated_corpus_w.csv")==False:
        print("abbreviated corpus creating.....")
        a=Abbreviations_Detection()
        a.Abbreviations(csv_file)
        print("abbreviation corpus created \t")
    else:
        print("abbreviated corpus file exists")
    

    if os.path.isfile('phrase_corpus_w.csv')==False:
        print("phrase corpus creating.....")
        generate_entities_and_labels_dictionary("abbreviated_corpus_w.csv")
        P=Phrase_Collection()
        sp,series=P.Phrases_with_hyphen('abbreviated_corpus_w.csv')
        P.Phrase_collection('abbreviated_corpus_w.csv',get_entities(),sp,scoring='npmi')
        print("phrase corpus created \t")
    else:
        print("phrase corpus file exists")
    
    csv_file='phrase_corpus_w.csv'
    

    if os.path.isfile("que_list_w.pickle")==False :
        print("questions and response list creating.....")
        get_que_ans(csv_file,'questions','responses')
        print("questions and response list created")
    else:
        print("questions and response list exist")
        
    if os.path.isfile('single_words_dict_w.pickle')==False:
        print("single word definitions generating")
        get_nouns_list()
        definations_w=get_definations_w(get_series())
        pickle_create('single_words_dict_w.pickle', definations_w)
        print("single word definitions generated \t")
    else:
        print("single word definitions file exists")
        
    if os.path.isfile("que_intents_w.pickle")==False:
        print("documents intents file creating...")
        get_intents()
        print("documents intents file created\t")
    else:
        print("documents intents file exists")

    if os.path.isfile("que_syn_dict_w.pickle")==False:
        print("synonyms file creating")
        get_words_synonyms()
        print("synonyms file created \t ")
    else:
        print("synonyms file exists")
        
    if os.path.exists('corpus_w.txt')==False:
        print("building corpus dictionary")
        build_corpus_dictionary(csv_file)
        print("corpus dictionary created \t ")
    else:
        print("corpus dictionary exists")
    
    if os.path.exists("que_tfidf_w.pickle")==False:
        print("tf idf file creating")
        tf_idf_words()
        print("tf idf file created")
    else:
        print("tf idf file exists")
        
    if os.path.exists("que_vectors_w.pickle")==False:
        print("word2vec file creating")
        word2vec_modeling()
        print(" word2vec file created \t ")
    else:
        print("word2vec file exists")
    
    if os.path.isfile("abbreviations_fullforms_dict_w.pickle")==False:
        print("abbreviations file creating......")
        abb_check=pd.read_pickle("abbreviations_w.pickle")
        get_fullforms(abb_check)
        fullform_list=pd.read_pickle("abbreviations_fullforms_dict_w.pickle")
        print("abbreviations fill created")
    else:
        print("abbreviations file exists")
        
    ed=timer()
    
    print("built chatbot successfully......!",(ed-st)/60)


# In[29]:


def start_chatbot(usertext):

    from datetime import datetime 
    
    que_list=pd.read_pickle("que_list_w.pickle")
    ans_list=pd.read_pickle("ans_list_w.pickle")
    que_intents=pd.read_pickle("que_intents_w.pickle")
    ans_intents=pd.read_pickle("ans_intents_w.pickle")
    que_intents_pos=pd.read_pickle("que_intents_pos_w.pickle")
    ans_intents_pos=pd.read_pickle("ans_intents_pos_w.pickle")
    que_distinct_words=pd.read_pickle("que_distinct_words_w.pickle")
    ans_distinct_words=pd.read_pickle("ans_distinct_words_w.pickle")
    que_syn_dict_index=pd.read_pickle("que_syn_dict_index_w.pickle")
    que_syn_dict=pd.read_pickle("que_syn_dict_w.pickle")
    que_tfidf=pd.read_pickle("que_tfidf_w.pickle")
    ans_tfidf=pd.read_pickle("ans_tfidf_w.pickle")
    que_vector=pd.read_pickle("que_vectors_w.pickle")
    ans_vector=pd.read_pickle("ans_vectors_w.pickle")
    saved_model_que_w=pd.read_pickle("saved_model_que_w")
    model_que= pickle.loads(saved_model_que_w)
    saved_model_ans_w=pd.read_pickle("saved_model_ans_w")
    model_ans= pickle.loads(saved_model_ans_w)
    abb_check=pd.read_pickle("abbreviations_w.pickle")
    phrases_list=pd.read_pickle("phrases_w.pickle")
    fullform_list=pd.read_pickle("abbreviations_fullforms_dict_w.pickle")
    single_word_dict=pd.read_pickle("single_words_dict_w.pickle")
    single_word_dict= dict((k.lower(), v) for k, v in single_word_dict.items())
    corpus_dict = enchant.PyPWL("corpus_w.txt")
    standard_dict = enchant.Dict("en_US")
    dic = pd.read_pickle("words_set_w.pickle")
    
    
    
    flag=True
    P=Phrase_Collection()
    print('BOT : Hey Hi! I am here to answer your queries. Type bye to end conversation')
    # while(flag==True):
    
    final_list=[]
    user_input=usertext
    user_input=spell_check(user_input,dic,standard_dict,corpus_dict)
    user_input=abbreviation_check(abb_check,user_input)
    user_input=P.replace_phrases(phrases_list,user_input)
    uq1=derive_intents(tokenize_words_pos([user_input]))
    if(user_input!='bye'):
        if(user_input in ['thanks','thank you','thankyou'] ):
            flag=False
            print("BOT : You are welcome..")
            return "BOT : You are welcome.." 
        else:
            if(greeting(user_input)!=None): #greeting
                print("BOT : "+greeting(user_input))
                return "BOT : "+greeting(user_input) 
            elif "full_form" in uq1[0]:
                return find_fullform(uq1[0],fullform_list)
            else:
                l1,fg,nd=get_intents_index_final(uq1,que_syn_dict_index,que_syn_dict)
                #print(l1)
                if len(l1)!=0:
                    if len(l1)==1:
                        print("BOT : "+ans_list[l1[0]])
                        return "BOT : "+ans_list[l1[0]]
                    else:
                        final_list=l1
                elif len(uq1[0])==1:
                    user=uq1[0]
                    return single_word_def(user[0],single_word_dict)
                else:
                    print("BOT : "+"insufficient data")
                    return "BOT : "+"insufficient data"
                if len(final_list)!=0:
                    nd=vectorized_dictionary(final_list,user_input,uq1,que_list,que_syn_dict,que_tfidf,model_que)
                    val_que,ind_que,ms_que=find_response_match_v2(nd,que_vector,que_list,ans_list)
                    if len(list(set(val_que)))==1:
                        print("BOT : ",val_que[0])
                        return "BOT : ",val_que[0]
                    else:
                        print("BOT : i might not be precise but i got some relevant information")
                        
                        for i in set(val_que):
                            print(i)
                        return "BOT : i might not be precise but i got some relevant information "
                if  bool(nd):
                    que_syn_dict_index={**que_syn_dict_index,**nd}

    else: #end conversation
        flag=False
        print("ROBO: Bye! take care..")
        return "ROBO: Bye! take care.." 
    pickle_create("que_syn_dict_index_w.pickle",que_syn_dict_index)


# In[30]:


def test_chatbot_database_questions():

    que_list=pd.read_pickle("que_list_w.pickle")
    ans_list=pd.read_pickle("ans_list_w.pickle")
    que_intents=pd.read_pickle("que_intents_w.pickle")
    ans_intents=pd.read_pickle("ans_intents_w.pickle")
    que_intents_pos=pd.read_pickle("que_intents_pos_w.pickle")
    ans_intents_pos=pd.read_pickle("ans_intents_pos_w.pickle")
    que_distinct_words=pd.read_pickle("que_distinct_words_w.pickle")
    ans_distinct_words=pd.read_pickle("ans_distinct_words_w.pickle")
    que_syn_dict_index=pd.read_pickle("que_syn_dict_index_w.pickle")
    que_syn_dict=pd.read_pickle("que_syn_dict_w.pickle")
    que_tfidf=pd.read_pickle("que_tfidf_w.pickle")
    ans_tfidf=pd.read_pickle("ans_tfidf_w.pickle")
    que_vector=pd.read_pickle("que_vectors_w.pickle")
    ans_vector=pd.read_pickle("ans_vectors_w.pickle")
    saved_model_que_w=pd.read_pickle("saved_model_que_w")
    model_que= pickle.loads(saved_model_que_w)
    saved_model_ans_w=pd.read_pickle("saved_model_ans_w")
    model_ans= pickle.loads(saved_model_ans_w)
    abb_check=pd.read_pickle("abbreviations_w.pickle")
    phrases_list=pd.read_pickle("phrases_w.pickle")
    fullform_list=pd.read_pickle("abbreviations_fullforms_dict_w.pickle")
    single_word_dict=pd.read_pickle("single_words_dict_w.pickle")
    single_word_dict= dict((k.lower(), v) for k, v in single_word_dict.items())
    corpus_dict = enchant.PyPWL("corpus_w.txt")
    standard_dict = enchant.Dict("en_US")
    dic = pd.read_pickle("words_set_w.pickle")
    
    
    crt=0
    wng=0
    wng1=[]
    
    flag=True
    P=Phrase_Collection()
    print('BOT : Hey Hi! I am here to answer your queries. Type bye to end conversation')
    st11= timer()
    for i in range(0,len(que_list)):
        
        
        final_list=[]
        user_input=que_list[i]
        print("Que",i+1,":",user_input)
        user_input=spell_check(user_input,dic,standard_dict,corpus_dict)
        user_input=abbreviation_check(abb_check,user_input)
        user_input=P.replace_phrases(phrases_list,user_input)
        uq1=derive_intents(tokenize_words_pos([user_input]))
       
        if(user_input!='bye'):
            if(user_input in ['thanks','thank you','thankyou'] ):
                flag=False
                print("BOT : You are welcome..")
                
            else:
                if(greeting(user_input)!=None): #greeting
                    print("BOT : "+greeting(user_input))
                    crt+=1
                elif "full_form" in uq1[0] or "fullform" in uq1[0]:
                    x=find_fullform(uq1[0],fullform_list)
                    if x==1:
                        crt+=1
                    else:
                        wng+=1
                        wng1.append(i)
                else:
                    l1,fg,nd=get_intents_index_final(uq1,que_syn_dict_index,que_syn_dict)
                    #print(l1)
                    if len(l1)!=0:
                        if len(l1)==1:
                            if ans_list[i]==ans_list[l1[0]]:
                                print("BOT : "+ans_list[l1[0]])
                                crt+=1
                            else:
                                print("BOT :")
                                print("Real ans: \n",ans_list[i])
                                print("Given ans: \n",ans_list[l1[0]])
                                wng+=1
                                wng1.append(i)
                                
                        else:
                            final_list=l1
        
                    elif len(uq1[0])==1:
                        user=uq1[0]
                        x=single_word_def(user[0],single_word_dict)
                        if x==1:
                            crt+=1
                        else:
                            wng+=1
                            wng1.append(i)
                    else:
                        print("BOT : "+"Insufficient Data")
                        wng+=1
                    if len(final_list)!=0:
                        nd=vectorized_dictionary(final_list,user_input,uq1,que_list,que_syn_dict,que_tfidf,model_que)
                        val_que,ind_que,ms_que=find_response_match_v2(nd,que_vector,que_list,ans_list)
                        if ans_list[i] in val_que:
                            crt+=1
                            print("BOT : ",ans_list[i])
                        else:
                            print("BOT :")
                            print("Real ans: \n",ans_list[i])
                            print("Given ans: \n")
                            for i in set(val_que):
                                print(i)
                            wng+=1
                            wng1.append(i)
                        

        else: #end conversation
            flag=False
            print("ROBO: Bye! take care..")
    ed1=timer()
    print("total time \t",ed1-st11)
    print(crt,wng)
    print("list of wrong answers",wng1)


# In[31]:


def test_chatbot_twisted_questions(csv_file,csv_file_test):
    
    get_que_ans_test(csv_file_test,"questions","responses")
    que_listt=pd.read_pickle("que_list_test_w.pickle")
    ans_listt=pd.read_pickle("ans_list_test_w.pickle")
    que_list=pd.read_pickle("que_list_db_w.pickle")
    ans_list=pd.read_pickle("ans_list_db_w.pickle")
    que_intents=pd.read_pickle("que_intents_w.pickle")    
    ans_intents=pd.read_pickle("ans_intents_w.pickle")
    que_intents_pos=pd.read_pickle("que_intents_pos_w.pickle")
    ans_intents_pos=pd.read_pickle("ans_intents_pos_w.pickle")
    que_distinct_words=pd.read_pickle("que_distinct_words_w.pickle")
    ans_distinct_words=pd.read_pickle("ans_distinct_words_w.pickle")
    que_syn_dict_index=pd.read_pickle("que_syn_dict_index_w.pickle")
    que_syn_dict=pd.read_pickle("que_syn_dict_w.pickle")
    que_tfidf=pd.read_pickle("que_tfidf_w.pickle")
    ans_tfidf=pd.read_pickle("ans_tfidf_w.pickle")
    que_vector=pd.read_pickle("que_vectors_w.pickle")
    ans_vector=pd.read_pickle("ans_vectors_w.pickle")
    saved_model_que_w=pd.read_pickle("saved_model_que_w")
    model_que= pickle.loads(saved_model_que_w)
    saved_model_ans_w=pd.read_pickle("saved_model_ans_w")
    model_ans= pickle.loads(saved_model_ans_w)
    abb_check=pd.read_pickle("abbreviations_w.pickle")
    phrases_list=pd.read_pickle("phrases_w.pickle")
    fullform_list=pd.read_pickle("abbreviations_fullforms_dict_w.pickle")
    single_word_dict=pd.read_pickle("single_words_dict_w.pickle")
    single_word_dict= dict((k.lower(), v) for k, v in single_word_dict.items())
    corpus_dict = enchant.PyPWL("corpus_w.txt")
    standard_dict = enchant.Dict("en_US")
    dic = pd.read_pickle("words_set_w.pickle")
    
    print("len",len(list(que_syn_dict_index.keys())))
    crt=0
    wng=0
    wng1=[]
    
    flag=True
    P=Phrase_Collection()
    print('BOT : Hey Hi! I am here to answer your queries. Type bye to end conversation')
    st11= timer()
    for i in range(0,len(que_listt)):
        
        
        final_list=[]
        user_input=que_listt[i]
        print("Que",i+1,":",user_input)
        user_input=spell_check(user_input,dic,standard_dict,corpus_dict)
        user_input=abbreviation_check(abb_check,user_input)
        user_input=P.replace_phrases(phrases_list,user_input)
        uq1=derive_intents(tokenize_words_pos([user_input]))
        
        if(user_input!='bye'):
            if(user_input in ['thanks','thank you','thankyou'] ):
                flag=False
                print("BOT : You are welcome..")
                
            else:
                if(greeting(user_input)!=None): #greeting
                    print("BOT : "+greeting(user_input))
                    crt+=1
                elif "full_form" in uq1[0] or "fullform" in uq1[0]:
                    x=find_fullform(uq1[0],fullform_list)
                    if x==1:
                        crt+=1
                    else:
                        wng+=1
                        wng1.append(i)
                else:
                    l1,fg,nd=get_intents_index_final(uq1,que_syn_dict_index,que_syn_dict)
                    if len(l1)!=0:
                        if len(l1)==1:
                            if ans_listt[i]==ans_list[l1[0]]:
                                print("BOT : "+ans_list[l1[0]])
                                crt+=1
                            else:
                                print("BOT :")
                                print("Real ans: \n",ans_listt[i])
                                print("Given ans: \n",ans_list[l1[0]])
                                wng+=1
                                wng1.append(i)
                        else:
                            final_list=l1
        
                    elif len(uq1[0])==1:
                        user=uq1[0]
                        x=single_word_def(user[0],single_word_dict)
                        if x==1:
                            crt+=1
                        else:
                            wng+=1
                            wng1.append(i)
                    else:
                        print("BOT : "+"insufficient data")
                        wng+=1
                        wng1.append(i)
                    if len(final_list)>=1:
                            nd=vectorized_dictionary(l1,user_input,uq1,que_list,que_syn_dict,que_tfidf,model_que)
                            val_que,ind_que,ms_que=find_response_match_v2(nd,que_vector,que_list,ans_list)
                            print(val_que)
                            if ans_listt[i] in val_que:
                                crt+=1
                                print("BOT :",ans_listt[i])
                            else:
                                print("BOT :")
                                print("Real ans: \n",ans_listt[i])
                                print("Given ans: \n")
                                for b in set(val_que):
                                    print(b)
                                wng+=1
                                wng1.append(i)
                    if  bool(nd):
                        que_syn_dict_index={**que_syn_dict_index,**nd}
        
        else: #end conversation
            flag=False
            print("ROBO: Bye! take care..")
    ed1=timer()
    print("total time \t",ed1-st11)
    print(crt,wng)
    pickle_create("que_syn_dict_index_w.pickle",que_syn_dict_index)
    print("list of wrong answers",wng1)


# # checking files and running chatbot

# In[32]:
def word2vec_run():

    global csv_file,csv_file_test
    csv_file="QuestionsResponsesISRO.csv"
    csv_file_test="test_questions.csv"


    # In[33]:


    create_chatbot(csv_file)


    # In[39]:


    


    # In[37]:


    test_chatbot_database_questions()


    # In[42]:


    # test_chatbot_twisted_questions(csv_file,csv_file_test)

    # while(True):
    #     print(start_chatbot())
    #     end1=int(input("enter 1 to end "))
    #     if(end1==1):
    #         break


# In[ ]:
# word2vec_run()




