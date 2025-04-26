
import streamlit as st
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
from keras.models import load_model
import time
import os
import librosa

# Title of the app
st.title('Audio Classification system')

# Load the pickled models and keras model
#pscaler = joblib.load('/content/scaler.pkl')
#pencoder = joblib.load('/content/encoder.pkl')
pmodel = load_model('/root/sound_datasets/urbansound8k/saved_models/audio_classification_cnn.keras')

# load the saved model
#with open("ANN_Model.pickle", "rb") as f:
#    ANN_Model = pickle.load(f)


uploaded_file=st.file_uploader("Choose an Audio file",
                               type=[".wav",".mp3"], 
                               accept_multiple_files=False) #"wave",".flac",

#Saving the browsed audio in our local
def Save_audio(upload_audio):
    try:
        if not os.path.exists("uploads"):
            os.makedirs("uploads")
        save_path = os.path.join(os.getcwd(), "uploads", upload_audio.name)
        with open(save_path, 'wb') as f:
            f.write(upload_audio.getbuffer())
        return save_path
    except Exception as e:
        print("Error saving file:", e)
        return None

#extract features using librosa(mfcc)
def extract_feature(file):
    data, sample_rates=librosa.load(file)
    mfcc_features=librosa.feature.mfcc(y=data,sr=sample_rates,n_mfcc=40)
    mfcc_scaled_feature=np.mean(mfcc_features.T,axis=0)

    mfcc_scaled_feature = mfcc_scaled_feature.reshape(1, -1)
    print(000,type(mfcc_scaled_feature),mfcc_scaled_feature)
    return mfcc_scaled_feature 


# Add a submit button
if st.button('Submit'):

  extract_features=[]
  if uploaded_file is not None:
      if Save_audio(uploaded_file):
          audio_bytes = uploaded_file.read()
          st.audio(audio_bytes, format="audio/wav")
          # extract_features.append(extract_feature(os.path.join("uploads",uploaded_file.name)))
          extract_features = extract_feature(os.path.join("uploads",uploaded_file.name))
          print(111,extract_features)
          progress_text = "Hold on! Result will shown below."
          my_bar = st.progress(0, text=progress_text)
          for percent_complete in range(100):
              time.sleep(0.02)
              my_bar.progress(percent_complete + 1, text=progress_text) ## to add progress bar untill feature got extracted


          # use the loaded model for prediction

          # Reshape the features
          # mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

          # Reshape the features to match the input shape of the CNN model
          #mfccs_extract_features = extract_features.reshape(1, extract_features.shape[0], 1) 
          # Reshape to (1, number of features, 1)

          #predictions = pmodel.predict(np.array(extract_features))
          predictions = pmodel.predict(extract_features)
          pred_class = np.argmax(predictions)

          # Map the predicted label index to the actual class label
          class_names = ['Air Conditioner', 'Car Horn', 'Children Playing', 'Dog Bark',
                        'Drilling', 'Engine Idling', 'Gun Shot', 'Jackhammer', 'Siren',
                        'Street Music']
          prediction_class = class_names[pred_class]

          print(prediction_class)

          #with open("Categories.pickle", "rb") as f:
          #    Categories = pickle.load(f)
          #class_cat=Categories[Categories['Class_ID']==pred_class]['Category']
          #shows the classified audio on page
          #bold_text = f"<t>{np.array(class_cat)[0]}</t>"
          bold_text = f"<t>{prediction_class}</t>"
          st.write(f'<span style="font-size:20px;">This Uploaded sound clip is {bold_text}</span>', unsafe_allow_html=True)
