import os
from google_auth_oauthlib.flow import InstalledAppFlow
import pickle
from datetime import datetime
from requests import api
from apiclient.discovery import build

from tqdm import tqdm
import requests
import json
import re
import time
import requests


Api_key = open("apikey.txt", 'r')
Api_key = str(Api_key.read())

import ast
# open finished labels json file
labels_file = open("finished-labels.json", "r")
labels_file_dump = labels_file.read()
# interpret the file into an array of dicts using ast.literal_eval
labels_arr = ast.literal_eval(labels_file_dump)





for i, video in tqdm(enumerate(labels_arr)):
    # print(video)
    # print(video["image"][video["image"].rfind("/") + 1:video["image"].rfind(".jpg")])
    video["id"] = video["image"][video["image"].rfind("/") + 1:video["image"].rfind(".jpg")]
    video["hq_default"] = "https://img.youtube.com/vi/" + video["id"] + "/hqdefault.jpg"
    print(video)
    headers = {

        "Authorization" : "Basic ",
        'Content-Type': 'application/json'
    }
    response = requests.post("https://search-mth420-project2-d6zohqo2oxsc5gsn23f5aavdkm.us-west-2.es.amazonaws.com/labels/_doc", data=json.dumps(video),headers=headers)
    print("\n\n")
    # print(response.status_code)
    # print(response.content)





# curl -X POST -u 