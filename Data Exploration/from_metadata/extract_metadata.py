import json 
import os
import shutil
import ast
import pandas as pd
import csv
import re


guests = []
guestbios = []
titles = []
ratings = []
lengths = []
views = []
text = ''

# find all podcast directories
podcasts = [x[0] for x in os.walk("./RawData/") if "podcast_" in x[0]]
for podcast in podcasts:
	for file in os.listdir(podcast):
		# find metadata file for this podcast
		if file.endswith(".txt") and not file.endswith("-cc.txt"):
			print('Processing file : ', file)

			# read metadata for this podcast
			with open(podcast+'/'+file, "r") as metadata:
				dict_metadata = ast.literal_eval(metadata.read())
				if dict_metadata:
					# parse guest details
					guests.append(dict_metadata['title'].split("_")[0])
					guestbios.append(dict_metadata['description'].split(".")[0])

					# parse podcast metadata
					titles.append(dict_metadata['title'])

					desc_start_idx = dict_metadata['description'].find("OUTLINE:")
					desc_end_idx = dict_metadata['description'].find("CONNECT:")
					segments = dict_metadata['description'][desc_start_idx+8:desc_end_idx-1].strip().split('\n')[1:]
					text += ''.join(segments)
					
					ratings.append(dict_metadata['rating'])
					lengths.append(dict_metadata['length'])
					views.append(dict_metadata['views'])

with open('topics.txt', 'w') as topics:
	text_cleaned = re.sub(r'http\S+', '', text)
	topic_text = text_cleaned.replace(\
						"SOCIAL:- Twitter:  LinkedIn:  Facebook:  Instagram:  Medium:  Reddit:  Support on Patreon:",\
						"")
	topics.write(topic_text)

with open('podcast_meta.csv', 'w') as podcast_meta:
	wr = csv.writer(podcast_meta, quoting=csv.QUOTE_ALL)
	wr.writerow(['titles', 'ratings', 'lengths', 'views'])
	for i in range(0, len(titles)):
		row = []
		row.append(titles[i])
		row.append(ratings[i])
		row.append(lengths[i])
		row.append(views[i])
		wr = csv.writer(podcast_meta, quoting=csv.QUOTE_ALL)
		wr.writerow(row)

with open('guest_meta.csv', 'w') as guest_meta:
	wr = csv.writer(guest_meta, quoting=csv.QUOTE_ALL)
	wr.writerow(['name', 'bio'])
	for i in range(0, len(titles)):
		row = []
		row.append(guests[i])
		row.append(guestbios[i])
		wr = csv.writer(guest_meta, quoting=csv.QUOTE_ALL)
		wr.writerow(row)