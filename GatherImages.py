import os
from google_auth_oauthlib.flow import InstalledAppFlow
import pickle
from datetime import datetime
from requests import api
from apiclient.discovery import build
from botocore.exceptions import NoCredentialsError
import boto3
from tqdm import tqdm
import requests
import json
import re
import time
import requests
import mimetypes
import urllib

credentials = r'C:\\Users\SplashFreeze\Desktop\Semester 7\\MTH 4312\\Project2\\aws_credentials.txt'
credentials = open(credentials, "r")
credentials = credentials.readlines()
ACCESS_KEY_ID = credentials[0].replace("\n", "")
SECRET_ACCESS_KEY = credentials[1].replace("\n", "")

# https://erangad.medium.com/upload-a-remote-image-to-s3-without-saving-it-first-with-python-def9c6ee1140
def upload_file(remote_url, bucket, file_name):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY_ID, aws_secret_access_key=SECRET_ACCESS_KEY)
    try:
        imageResponse = requests.get(remote_url, stream=True).raw
        
        content_type = imageResponse.headers['content-type']
        # print("Content_type is ", content_type)
        # print("Remote URL is ", remote_url)
        
        extension = mimetypes.guess_extension(content_type)
        # print("File is ", file_name + extension)
        urllib.request.urlretrieve(remote_url, file_name + extension)
        # print("Extension is ", extension)
        # print("Uploading file name is ", file_name[file_name.rfind("\\")+1:] + extension)
        s3.upload_file(file_name + extension, bucket,file_name[file_name.rfind("\\")+1:] + extension)
        # print(remote_url)
        # print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False
# uploaded = upload_file('url_of_the_file_to_be_uploaded', 'bucket-name', 'file_name')

# # curl -X POST -u 



channels = os.listdir(".\channels")
print(channels)


# Windows Version:
directory = r'C:\\Users\SplashFreeze\Desktop\Semester 7\\MTH 4312\\Project2\\channels'
# directory = '/home/davidn/Documents/MTH-5320/Project1/channels'



failed_channels = []
parallel = []
iteratorLines = []

if(os.path.exists(".\iterator.txt")):
    iteratorContent = open(".\\iterator.txt", "r")
    iteratorLines = iteratorContent.readlines()

channelStartIndx = 0
videoStartIndx = 0

if(len(iteratorLines) != 0):
    channelStartIndx = channels.index(iteratorLines[0].replace("\n", ""))
    o = open(os.path.join(directory, iteratorLines[0].replace("\n", "")) + "\\videos.txt","r")
    videoIDs = o.readlines()
    videoIDs = [x.replace("\n","") for x in videoIDs]
    # print("VideoIDs are ",videoIDs)
    videoStartIndx = videoIDs.index(iteratorLines[1].replace("\n", ""))
    print("Channel Start Index is ", channelStartIndx)
    print("Video Start Index is ", videoStartIndx)
else:
    print("Iterator is 0\n Starting form 0!\n")

print("uploading file!!")

if(channelStartIndx != 0):
    channels = channels[channelStartIndx:]
for channel in tqdm(channels):
    # print("Channel is ", channel)
    iterator = open(".\iterator.txt", "w")
    iterator.truncate(0)
    iterator.write(channel)

    o = open(os.path.join(directory, channel) + "\\videos.txt","r")
    videoIDs = o.readlines()
    videoIDs = [x.replace("\n","") for x in videoIDs]

    if(channel == channels[channelStartIndx]):
        # print("Starting at index ", videoStartIndx)
        videoIDs = videoIDs[videoStartIndx:]

    for count, videoID in enumerate(videoIDs):
            iterator = open(".\iterator.txt", "a")
            iterator.truncate(0)
            iterator.write(channel + "\n")
            iterator.write(videoID + "\n")
            upload_file("https://img.youtube.com/vi/" + videoID + "/hqdefault.jpg", "mth5320-thumbnails", ".\\channels\\" + channel + "\\" + videoID)
            # print(videoID)
            # to get the thumbnail of a video, you do:
            # https://img.youtube.com/vi/<insert-youtube-video-id-here>/hqdefault.jpg



