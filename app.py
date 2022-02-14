
from flask import Flask, render_template, request, redirect, url_for,session
from tfidf import *
from word2vec import *
# from Synonym_context_chatbot import SCC, own_func, startbot
from Synonym_context_chatbot import *


app = Flask(__name__)
app.secret_key = "any random string"

@app.route('/')
def index():
    return render_template('ChatBot.html')

@app.route('/upload_file1', methods = ['GET','POST'])
def upload_file1():
    uploaded_file1 = request.files['file1']
    uploaded_file2 = request.files['file']
    print(uploaded_file1)
    if uploaded_file1.filename != '' :
        uploaded_file1.save(uploaded_file1.filename)
    if uploaded_file2.filename != '':
        uploaded_file2.save(uploaded_file2.filename)
        print(request.form.get("file1"))
        # session['uploaded_file1']=uploaded_file1
        session['first_file'] =request.form.get("file1")
        session['Second_file'] =request.form.get("file")
        print("filename",session['first_file'],session['Second_file'])
    return redirect(url_for('index'))
    
@app.route('/typebot', methods = ['GET','POST'])
def typebot():
    s = request.args.get("types")
    print("sssss",s)
    # tfidf_run()
    # s=request.form.get("bottype")
    session["s"] = s
    # print("sssss",s)
    # oc=session['first_file']
    # tq=session['Second_file']
    # print("occc,tccc",oc,tq)
    if s == "tfidf":
        tfidf_run()
    elif s =="synonym":
        own_func()
    elif s == "word2vec":
        word2vec_run()
    # # scc = own_func(upload_file1,upload_file2)


    return render_template('index.html')
@app.route('/feedback', methods = ['GET','POST'])

def feedback():
    return render_template('feed1.html')
@app.route('/indexfile', methods = ['GET','POST'])

def indexfile():
    return render_template('index.html')
@app.route('/reviewbot', methods = ['GET','POST'])
def reviewbot():
    resp = request.form.get("botreview")
    print(resp,"resp")
    if resp == "yes":
        return render_template('ChatBot.html')
        
    else:
        return render_template('feedback.html',flag=2)


@app.route('/userfb', methods = ['GET','POST'])
def userfb():
    botres = request.form.get("feedback_res")
    print("botres",botres)
    return render_template('feedback.html',flag=3)
    # if botres:
    #     print("Back to home page")
    # else:
    #     Flask.s.clear()


@app.route('/userresponse', methods = ['GET','POST'])
def userresponse():
    s=request.form.get("userresponse")
    print("userresponse",s)
    return redirect(url_for('typebot'))
@app.route("/get")
def get_bot_response():
    userText = request.args.get("msg") #get data from input,we write js  to index.html
    print("texttt",userText)
    # res = run_chatbot(userText)
    s = session["s"]
    if s == "tfidf":
        res = run_chatbot(userText)
    elif s =="synonym":
        res=startbot(userText)
    elif s == "word2vec":
        res=start_chatbot(userText)
    # StartChatbot(qn_intents,res_intents,2,word_to_qn_intents_dict,word_to_res_intents_dict,questionlist,responselist,0.5,2,5,syn_dict)

    

    
    # userText1=userText.lower()
    # if userText1=="bye":
    #     return redirect(url_for('feedback'))
        # return render_template('feedback.html',flag=1)

    # x=startbot(userText)
    # x=run_chatbot(userText)
    print("xxxxxxx",res)
    return res


if __name__ == "__main__":
     app.run(debug = True)

    