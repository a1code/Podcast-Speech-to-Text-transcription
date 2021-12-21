from pydub import AudioSegment
import json
import os
import string

# Main function 
def main():
	# find all podcast directories
	podcasts = [x[0] for x in os.walk("./RawData/") if "podcast_" in x[0]]
	for podcast in podcasts:
		foldername = podcast.split("/")[-1]
		os.makedirs('./PreparedData/'+foldername)
		for file in os.listdir(podcast):
			# find podcast audio file in this directory
			if file.endswith(".wav"):
				print('Processing file : ', file)

				filename = file.replace(".wav", "")

				# read wav audio file for this podcast
				audio = AudioSegment.from_wav(podcast+'/'+file)

				# read closed captions for this podcast
				with open(podcast+'/'+filename+"-cc.txt", "r") as cc:
					cc_obj = json.loads(cc.read())

					# create splits
					segment_timestamps = []
					for i in range(0, len(cc_obj)-1):
						segment = [cc_obj[i]['start'], cc_obj[i+1]['start']+0.01]
						segment_timestamps.append(segment)

					segment_transcripts = [x['text'].translate(str.maketrans('', '', string.punctuation))\
					.replace('\n', ' ') for x in cc_obj]

					# process each split
					for idx, seg in enumerate(segment_timestamps):
						audio_chunk = audio[seg[0]*1000:seg[1]*1000]	# pydub works in milliseconds
						text_chunk = segment_transcripts[idx]
						# ignore splits that correspond to just one word in the transcript
						if len(text_chunk.split()) > 4:
							# write split to the directory for this podcast
							audio_chunk.export('./PreparedData/'+foldername+'/'\
								+filename+"#"\
								+str(seg[0])+"#"\
								+str(seg[1])\
								+".wav", format="wav")
							with open('./PreparedData/'+foldername+'/'\
								+filename+"#"\
								+str(seg[0])+"#"\
								+str(seg[1])\
								+".txt", 'w') as f:
								f.write(text_chunk)

# execution starts here 
if __name__=="__main__": 
	main() 