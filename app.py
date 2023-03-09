import os
import time
import pandas as pd
import openai
import re
import requests
import sys
from num2words import num2words
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
import redis
import tiktoken
import os

import pickle
import flask
import os
import zlib
from flask import Response
from flask import Flask, request, jsonify
from flask.logging import create_logger
import logging

#test
app = flask.Flask(__name__)
port = int(os.getenv("PORT", 9099))
##api = Api(app)
logging.basicConfig(filename='demo.log', level=logging.DEBUG)


openai.api_type = "azure"
openai.api_key = "228e404c5ace452880b0c4ed50bf40c5"
openai.api_base = "https://cstoai-eastus.openai.azure.com/"
openai.api_version = "2022-12-01"

##Redis Data
redis_host = 'gptstore.redis.cache.windows.net'
redis_port = 6380
redis_db = 0
redis_key= "ZRrwentVPr4V2dSGMZeW18UxJn1QB3ptxAzCaGzLBh0="
#redis_ssl = 'true'
redis_client = redis.Redis(host=redis_host, port=redis_port, password=redis_key, ssl=True)



def getdata(redis_client):
    redis_client.keys()
    rehydrated_df = pickle.loads(zlib.decompress(redis_client.get('response')))
    df_final= pd.DataFrame(rehydrated_df)
    return df_final

def rate_limited(func):
    min_wait = 5 / 10 # 5 requests per second
    last_time_called = [time.monotonic()]

    def wrapper(*args, **kwargs):
        elapsed_time = time.monotonic() - last_time_called[0]
        time_to_wait = max(min_wait - elapsed_time, 0)
        time.sleep(time_to_wait)
        result = func(*args, **kwargs)
        last_time_called[0] = time.monotonic()
        return result

    return wrapper

@rate_limited
def generate_embeddings(word):
    try:
        return openai.Embedding.create(input = [text], engine="text-embedding-ada-002")['data'][0]['embedding'] 
    except:
        return np.zeros(768)
    
def search_docs(df, user_query, top_n=2, to_print=True):
    embedding = generate_embeddings(user_query)
    app.logger.info("Genrated embeddings")
    #print(embedding.shape)

    df["similarities"] = df.ada_v2_embedding.apply(lambda x: cosine_similarity(x, embedding))

    app.logger.info("similarity checkkkkk")

    res = (
        df.sort_values("similarities", ascending=False)
        .head(top_n)
    )
    return res



#@app.route('/predict', methods=['POST'])
def predict(ai_question,redis_client):
    app.logger.info("enter prediction111111111111")
    logging.info("enter prediction111111111111")
    #ai_question = flask.request.get_json(force=True)
    datadf= getdata(redis_client)
    app.logger.info("enter prediction")
    logging.info("enter datadf")
    #LOG.info("JSON payload: %s datadf")
    #app.logger.info(datadf)
    question = ai_question["ai_question"]
    app.logger.info(question)
    print(question)
    res = search_docs(datadf, question, top_n=2)
    app.logger.info(res)
    context= res.Content.values
    completion_model='tsg'
    initial_prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly."
    combined_prompt = initial_prompt + str(context) + "Q: " + question
    response = openai.Completion.create(engine=completion_model, prompt=combined_prompt, max_tokens=500)
    ai_response = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
    response = {"ai_response":ai_response}
    app.logger.info(ai_response)
    return jsonify(ai_response)

@app.route('/foo', methods=['POST']) 
def foo():
    app.logger.info("Enter Mai!!!!n")
    logging.info("Enter Mai!!!!n----------> foo ")
    app.logger.info('Processing default request')
    logging.info("rocessing default request")
    ai_question = flask.request.get_json(force=True)
    app.logger.info('ai_question')
    logging.info("ai_question")
    #redis_client = redis.Redis(host=redis_host, port=redis_port, password=redis_key, ssl=True)
    response = predict(ai_question,redis_client)
    return response

'''@app.route("/predict", methods=['POST'])
def get_bot_response():
    ai_question = flask.request.get_json(force=True)
    #userText = request.args.get('msg')
    return  predict(ai_question)'''

'''
@app.route("/")
def get():
    app.logger.info("enter /////")
    return "djopjdow"

'''
    
if __name__ == "__main__":
    app.logger.info("Enter Main")
    logging.info("Enterrrrr")
    app.run(port=8080)
