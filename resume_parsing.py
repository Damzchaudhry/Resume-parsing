#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.layout import LAParams
import io
from nltk.tokenize import sent_tokenize


def pdfparser(data):

    fp = open('resume/dj.pdf', 'rb')
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    # Process each page contained in the document.

    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
        data =  retstr.getvalue()
    
    
    sents = sent_tokenize(data)
    return sents,data
    

[token,string]=pdfparser(sys.argv[1])  


# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:


import spacy
from spacy.matcher import Matcher

# load pre-trained model
nlp = spacy.load('en_core_web_sm')


# In[3]:


import re

def extract_mobile_number(text):
    phone = re.findall(re.compile(r'(?:(?:\+?([1-9]|[0-9][0-9]|[0-9][0-9][0-9])\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([0-9][1-9]|[0-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?'), text)
    
    if phone:
        number = ''.join(phone[0])
        if len(number) > 10:
            return '+' + number
        else:
            return number


# In[4]:


import re

def extract_email(email):
    email = re.findall("([^@|\s]+@[^@]+\.[^@|\s]+)", email)
    if email:
        try:
            return email[0].split()[0].strip(';')
        except IndexError:
            return None


# In[5]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer("[a-zA-z@]+")
sw = set(stopwords.words ('english'))
def tokenn(sentence):
    def filter_words(word_list):
        useful_words=[w for w in word_list if w not in sw ]
        return useful_words
    
    
    def myTokenizer(sentence):
            words = tokenizer.tokenize(sentence.lower())
            return filter_words(words)
    
    cv=CountVectorizer(tokenizer = myTokenizer,ngram_range=(1,4))
    vectorized_corpus = cv.fit_transform(sentence)
    vc=vectorized_corpus.toarray()
    fl=list(cv.inverse_transform(vc))
    return fl
    
    


# In[6]:


def extract_skills(sentence):
            fl=tokenn(sentence)
            data = pd.read_csv("skills.csv") 
            skills = list(data.columns.values)
            skillset = []

            for i in range(len(fl)):
                for j in skills:
                    if j  in fl[i]: 
                        skillset.append(j)

            return skillset


# In[7]:


ex=extract_skills(token)
ex=list(set(ex))
ex


# In[8]:


extract_mobile_number(string) 


# In[9]:


email = extract_email(string)
email


# In[10]:


print(string)


# In[11]:


# string.split(" ")


# In[ ]:





# In[12]:


def extract_name(string):
    r1 = str(string)
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(r1)
    for ent in doc.ents:
        if(ent.label_ == 'PERSON'):
            print(ent.text)
            break


# In[13]:



import glob
import pandas as pd

# get data file names
path =r'./indian-names'
filenames = glob.glob(path + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))

# Concatenate all data into one DataFrame
big_frame = pd.concat(dfs, ignore_index=True)


# In[14]:


big_frame.head()


# In[15]:


(extract_name(string))


# In[16]:


dff=big_frame['name'].values


# In[17]:


dff


# In[18]:


def extract_name1(sentence):
            fl=tokenn(sentence) 
            names = list(dff)
            names_set = []

            for i in range(len(fl)):
                for j in names:
                    if j  in fl[i]: 
                        names_set.append(j)

            return list(set(names_set))


# In[ ]:





# In[19]:


res=""
pos=email.find('@')
new=email[:pos]
new=re.findall("[a-zA-Z]",new)
res="".join(new)

if(res!=""):
    try:
        
        if(len(extract_name1(token))>1):
                for i in extract_name1(token):
                        if i in email:
                            res=(i)
            
        else:
            res= extract_name1(token)[0]
                    
           
    except:
          res=extract_name(string)
#else:
#     pos=email.find('@')
#     new=email[:pos]
#     new=re.findall("[a-zA-Z]",new)
#     res="".join(new)


# In[20]:


extract_name1(token)


# In[21]:


res


# In[35]:





# In[36]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




