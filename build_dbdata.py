import chromadb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import fireworks.client
import os
import hashlib

#setting
dataset_filename = 'chat_dataset_eng.json'
time_dif = timedelta(minutes=60 * 12)

def generate_uid(input_string):
    # Create a SHA-256 hash object
    hash_object = hashlib.sha256()
    
    # Update the hash object with the input string
    hash_object.update(input_string.encode())
    
    # Generate the hexadecimal representation of the hash
    uid = hash_object.hexdigest()
    
    return uid

with open(dataset_filename) as f:
    chat_all = json.load(f)

#print(chat_all)
#print(len(chat_all))

past_time = ''
conversations = []
for i in range(len(chat_all)):
    #2023-10-07 08:39:00
    single_time = datetime.strptime(chat_all[i]['time'], '%Y-%m-%d %H:%M:%S') 
    if past_time == '':
        conversation = []
        past_time = single_time
        conversation.append(chat_all[i])
        continue

    if (single_time - past_time) > time_dif:
        conversations.append(conversation)
        conversation = []
        conversation.append(chat_all[i])
        past_time = single_time
    else:
        conversation.append(chat_all[i])
        past_time = single_time

def get_completion(prompt, model=None, max_tokens=100):

    fw_model_dir = "accounts/fireworks/models/"

    if model is None:
        model = fw_model_dir + "llama-v2-7b"
    else:
        model = fw_model_dir + model

    completion = fireworks.client.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0
    )

    return completion.choices[0].text

print('cek total conversation', len(conversations))
for i in range(len(conversations)):
    print('conv %s len %s'%(i+1, len(conversations[i])))

#exit()
default_mood = 'flat'
default_event = 'none'

# Set the environment variable
fireworks.client.api_key = os.getenv("FIREWORK_KEY")
mistral_llm = "mistral-7b-instruct-4k"

client = chromadb.PersistentClient(path="./personality_bot2")

# create collection
collection = client.get_or_create_collection(
    name=f"memorybank"
)

metadatas = []
documents = []
ids = []
for i in range(len(conversations)):
    #if i != 66:
    #    continue
    conversation = ''
    for j in range(len(conversations[i])):
        single_dialog = '%s : %s\n'%(conversations[i][j]['user'], conversations[i][j]['chat'])
        conversation += single_dialog
    print(conversation)
    conversation_id = generate_uid(conversation+str(i))

    #get event
    prompt_event = '[INST]Summarize the event in this conversation below as short as possible\n %s[/INST]'%(conversation)
    single_event = get_completion(prompt_event, model=mistral_llm)
    single_event_id = generate_uid(single_event+str(i))

    prompt_mood = '[INST]Based on the following dialogue, please provide a highly concise summary of the Namine personality traits and emotions\n %s[/INST]'%(conversation)
    single_mood = get_completion(prompt_mood, model=mistral_llm)
    #print(single_mood)
    #exit()
    single_mood_id = generate_uid(single_mood+str(i))
    #print('MOOD: ', single_mood)

    #single_time_begin = datetime.strptime(conversations[i][0]['time'], '%Y-%m-%d %H:%M:%S')
    #single_time_end = datetime.strptime(conversations[i][-1]['time'], '%Y-%m-%d %H:%M:%S')

    single_time_begin = conversations[i][0]['time']
    single_time_end = conversations[i][-1]['time']

    metadata_conversation = {}
    metadata_conversation['type'] = 'conversation'
    metadata_conversation['begin time'] = single_time_begin
    metadata_conversation['end time'] = single_time_end

    metadata_mood = {}
    metadata_mood['type'] = 'mood'
    metadata_mood['begin time'] = single_time_begin
    metadata_mood['end time'] = single_time_end

    metadata_event = {}
    metadata_event['type'] = 'event'
    metadata_event['begin time'] = single_time_begin
    metadata_event['end time'] = single_time_end

    documents.append(conversation)
    documents.append(single_event)
    documents.append(single_mood)

    metadatas.append(metadata_conversation)
    metadatas.append(metadata_event)
    metadatas.append(metadata_mood)
    
    ids.append(conversation_id)
    ids.append(single_event_id)
    ids.append(single_mood_id)

    print(single_time_begin, single_time_end)

    #break

collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids)

#try:
#    collection.add(
#        documents=single_review,
#        metadatas=single_metadata,
#        ids=single_review_uid
#    )
#except chromadb.errors.DuplicateIDError:
#    print('already add')
#except chromadb.errors.IDAlreadyExistsError as id_error:
#    print(f"already add => {id_error}")