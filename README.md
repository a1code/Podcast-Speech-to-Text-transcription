# Podcast-Speech-to-Text-transcription

**Note**: Please watch the final presentation [here](https://drive.google.com/file/d/1AueeMUUUz2xrMubkompdtnaVOo_3VWmx/view?usp=sharing) or refer [STTProjectDescription.pdf](https://github.com/a1code/Podcast-Speech-to-Text-transcription/blob/main/STT_Project_Description.pdf) for details.

**Dataset** : [Lex Fridman podcast](https://www.youtube.com/watch?v=Z_LhPMhkEdw&list=PLrAXtmErZgOdP_8GztsuKi9nrraNbKKp4).  
The repository contains all the scripts to download and setup the dataset from scratch. However, the pre-processed dataset from our implementation can be found [here](https://drive.google.com/file/d/1oaeXv8U2Z8We2m0FJEreked_GxjAZK_F/view?usp=sharing).  
The raw data comprises of audio episodes (RIFF little-endian data, WAVE audio, Microsoft PCM, 16 bit, mono 16000 Hz) from the playlist, along with the corresponding video descriptions as metadata, and English closed captions as ground truth for model building. After preparation, the dataset contains (X, Y) pairs where X is an audio segment of length >= 4 seconds, and Y is the corresponding text transcript.

**Implementation Summary**:  
• Implemented an end-to-end data science workflow to collect, prepare, explore and transcribe audios to text for Lex Fridman podcasts.  
• Wrote modules for audio feature extraction and the DeepSpeech2 Encoder-Decoder network in PyTorch, including training, tuning and evaluation of the model using around 600 hours of speech.  
• Achieved a final Character Error Rate of ~35%, and set up a Python module to reuse trained weights for inference over new data.    

**Using the pre-trained model**:  
Requirements -  
python (>=3)  
torch (>=1.9)  
ffmpeg  
pytube  
pydub

To make inference using the pre-trained weights, download the Deployment folder in the repository as seen [here](https://github.com/a1code/Podcast-Speech-to-Text-transcription/tree/main/Deployment). Then run the below commands to generate the text transcript file for the Youtube video.  
```
cd Deployment/
python3 transcription.py [youtube_url]
```

For consistency, we only tested this on a new episode of Lex Fridman podcast, specifically the episode [here](https://www.youtube.com/watch?v=nDDJFvuFXdc).

