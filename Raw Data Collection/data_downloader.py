import os
import subprocess
from pytube import Playlist
import re
import json
import time
from datetime import datetime, timezone
from youtube_transcript_api import YouTubeTranscriptApi


def delete_file_from_path(path):
	try:
		os.remove(path)
	except OSError as e:
		print("Error: %s : %s" % (path, e.strerror))


# Main function 
def main():
	# playlist to download the podcasts from
	playlist_link = "https://www.youtube.com/watch?v=YJF01_ztxwY&list=PLrAXtmErZgOdP_8GztsuKi9nrraNbKKp4"
	playlist = Playlist(playlist_link)
	# this fixes the empty playlist.videos list
	playlist._video_regex = re.compile(r"\"url\":\"(/watch\?v=[\w-]*)")
	print('Total number of podcasts in the playlist: ', str(len(playlist.video_urls)))

	for video in playlist.videos:
		try:
			# get metadata for this podcast
			metadata = {}
			metadata['video_id'] = video.video_id
			metadata['title'] = re.sub('[^a-zA-Z0-9 \n\.]', '_', video.title)
			metadata['description'] = video.description
			metadata['rating'] = video.rating
			metadata['length'] = video.length
			metadata['views'] = video.views
			metadata['author'] = video.author
			aware_local_now = datetime.now(timezone.utc).astimezone()
			metadata['downloaded_at'] = str(aware_local_now)

			# get closed captions for this podcast
			cc = YouTubeTranscriptApi.get_transcript(metadata['video_id'], languages=['en'])
		except Exception:
			continue

		# create directory for this podcast
		download_dir = 'podcast_'+metadata['video_id']
		os.makedirs(download_dir, exist_ok=True)

		# open a new text file to write metadata for this podcast
		with open(os.path.join(download_dir, metadata['title']+'.txt'),'w+') as f:
			# append metadata to text file
			f.write(str(metadata) + '\n')

		# open a new text file to write closed captions for this podcast
		with open(os.path.join(download_dir, metadata['title']+'-cc.txt'),'w+') as f:
			f.write(json.dumps(cc))

		# start audio mp4 download stream for this podcast
		ys = video.streams.filter(only_audio=True, file_extension='mp4')
		print('Downloading : ', metadata['title'])
		ys[0].download(filename=metadata['title']+'.mp4')

		# to ensure mp4 is fully downloaded before further processing
		time.sleep(2)

		'''
		convert downloaded mp4 to wav file 
		RIFF (little-endian) data, WAVE audio, Microsoft PCM, 16 bit, mono 16000 Hz
		and save in the directory created for this podcast
		'''
		subprocess.call(['ffmpeg', \
			'-i', metadata['title']+'.mp4', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', \
			os.path.join(download_dir, metadata['title']+'.wav')])

		# delete raw mp4 file
		delete_file_from_path(metadata['title']+'.mp4')


# execution starts here 
if __name__=="__main__": 
	main() 