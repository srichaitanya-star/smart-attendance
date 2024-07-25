import pickle
import cv2
import face_recognition
import face_recognition
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("serviceKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://attendencesystem-5f7d5-default-rtdb.firebaseio.com/",
    'storageBucket':"attendencesystem-5f7d5.appspot.com"
})



folderPath="Images"
PathList=os.listdir(folderPath)
imgList=[]
# print(modePathList)
# imgModeList=[]
studentIds=[]
# print(PathList)
for path in PathList:
    imgList.append(cv2.imread(os.path.join(folderPath,path)))
    # print(os.path.splitext(path)[0])
    studentIds.append(os.path.splitext(path)[0])


    fileName=f'{folderPath}/{path}'
    bucket=storage.bucket()
    blob=bucket.blob(fileName)
    blob.upload_from_filename(fileName)
# print(studentIds)
print("Encoding Started......")
def findEncodings(imagesList):
    encodeList=[]
    for img in imagesList:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
encodeListKnown=findEncodings(imgList)
encodeListKnownWithIds=[encodeListKnown,studentIds]
# print(encodeListKnown)
print("Encoding Complete.....")

file=open("EncodeFile.p",'wb')
pickle.dump(encodeListKnownWithIds,file)
file.close()
print("file saved")

