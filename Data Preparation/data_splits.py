import json
import os
import string
import random
from shutil import copyfile

# Main function 
def main():
	# find all podcast directories
	podcasts = [x[0] for x in os.walk("./PreparedData/") if "podcast_" in x[0]]
	for podcast in podcasts:
		foldername = podcast.split("/")[-1]
		print('Processing episode : ', foldername)

		# create folder for this episode in train/VALIDATION/test folder
		os.makedirs('./DataSplits/TRAIN/'+foldername)
		os.makedirs('./DataSplits/VALIDATION/'+foldername)
		os.makedirs('./DataSplits/TEST/'+foldername)

		# get all segments for this episode
		files = os.listdir(podcast)
		segments = [x.replace(".wav", "") for x in files if x.endswith(".wav")]

		# distribute segments randomly as 80% train, 10% VALIDATION, 10% test
		random.shuffle(segments)
		train = int(0.8 * len(segments))
		VALIDATION = int(0.1 * len(segments))
		train_set = segments[0:train+1]
		VALIDATION_set = segments[train+1:train+VALIDATION+1]
		test_set = segments[train+VALIDATION+1:]
		
		for record in train_set:
			copyfile(podcast+'/'+record+'.wav', './DataSplits/TRAIN/'+foldername+'/'+record+'.wav')
			copyfile(podcast+'/'+record+'.txt', './DataSplits/TRAIN/'+foldername+'/'+record+'.txt')
		for record in VALIDATION_set:
			copyfile(podcast+'/'+record+'.wav', './DataSplits/VALIDATION/'+foldername+'/'+record+'.wav')
			copyfile(podcast+'/'+record+'.txt', './DataSplits/VALIDATION/'+foldername+'/'+record+'.txt')
		for record in test_set:
			copyfile(podcast+'/'+record+'.wav', './DataSplits/TEST/'+foldername+'/'+record+'.wav')
			copyfile(podcast+'/'+record+'.txt', './DataSplits/TEST/'+foldername+'/'+record+'.txt')
		

# execution starts here 
if __name__=="__main__": 
	main() 