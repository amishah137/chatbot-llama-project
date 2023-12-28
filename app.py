#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 08:22:18 2023

@author: ami
"""
from flask import Flask, render_template, request

import pandas as pd
from langchain.schema import Document
from langchain.embeddings.sentence_transformer import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain import LLMChain

path = "myDataset.csv"
persist_directory = "./vector_db"


app = Flask(__name__)

@app.route('/sync', methods=['POST'])
def saveData():
    model = 'flax-sentence-embeddings/all_datasets_v3_roberta-large'

    docs = []
    data = pd.read_csv(path)

    for i in range(data.shape[0]):
        d = data.iloc[i]
        print(d['Name'])
        page_content = f"Employee {d['Name']} has experience of {d['Experience']} years in the role of {d['Role']} with skills of {d['Technology']}." 
        metadata = dict({"name":d['Name'], "role":d['Role'], 
                         "technology": d['Technology'],
                         "experience": d['Experience'], "source":path})
        docs.append(Document(page_content=page_content, metadata=metadata))
        docs = filter_complex_metadata(docs)

    with open("vectordb.txt","w+") as f:
        for items in docs:
           f.write('%s\n' %items)
    f.close()

    embedding_function = HuggingFaceEmbeddings(model_name=model)
    db = Chroma.from_documents(docs, embedding_function, persist_directory=persist_directory)
    print("Data Sync Successfully")
    return render_template('form.html')

def runChat(query):
    metadata_field_info = [
        AttributeInfo(
            name="name",
            description="Name of the employee",
            type="string",
        ),
        AttributeInfo(
            name="role",
            description="Designation of the employee",
            type="string",
        ),
        AttributeInfo(
            name="technology",
            description="Skillset of the employee",
            type="string",
        ),
        AttributeInfo(
            name="experience",
            description="Years of experience employee has in the role and technology",
            type="int or string",
        )
    ]
    
    template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "To assist you in finding information about employee skillset, recommend the list of employees with combined skillset to fulfill the requirement. The model is designed to handle variations in the input, including spelling mistakes, special characters, and different case characters. I will do my best to locate the neccasary information based on the context. The answer should be short and succinct. if you don't have context, you can say you don't know."
                )
            ),
            HumanMessagePromptTemplate.from_template("{ret}"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )
    
    
    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    persist_directory = "./vector_db"
    model = "flax-sentence-embeddings/all_datasets_v3_roberta-large"
    embedding_function = HuggingFaceEmbeddings(model_name=model)
    
    llm = LlamaCpp(
      model_path="models/llama-2-13b-chat.ggmlv3.q4_0.bin",
      n_ctx=2048,
      #n_gqa=8,
      n_gpu_layers=24,
      n_threads=4,
      n_batch=512,
      temperature=0,
      max_tokens=2000,
      top_p=0,
      callback_manager=callback_manager,
      verbose=False
    )
    
    # Now we can load the persisted database from disk, and use it as normal. 
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
    retriever = db.as_retriever()
    qa = LLMChain(prompt=template, llm=llm)
    retrieved_docs = retriever.get_relevant_documents(query)
    res = qa.run({'input':query,'ret': retrieved_docs})
    print(res)
    return res 


@app.route('/')
def welcome():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def result():
    query = request.form.get("Query", type=str, default=0)
    entry = runChat(query)
    return render_template('form.html', entry=entry)

if __name__ == '__main__':
    app.run(debug=True)