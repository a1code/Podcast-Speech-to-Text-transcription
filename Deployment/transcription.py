from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import math
import random
import numpy as np
import io
import sys
import subprocess
from pytube import YouTube
from pydub import AudioSegment
from pydub.utils import make_chunks

# CONFIGS
MODEL_CHECKPOINT = './checkpoint.pt'
batch_size = 32
workers = 2
MAX_LEN = 105

# VOCABULARY
chars = ' abcdefghijklmnopqrstuvwxyz'
vocab = ['<blank>', '<pad>', '<unk>']
for ch in chars:
  vocab.append(ch)
VOCAB_LEN = len(vocab)

class Encoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80)

	def forward(self, input_X):
		X_len = []
		AUDIO = []
		for X in input_X:
		  waveform, sample_rate = torchaudio.load(X)
		  audio_tensor = self.spec(waveform).squeeze(0).transpose(1, 0)
		  # print('spectrogram', audio_tensor.shape)
		  AUDIO.append(audio_tensor)
		  X_len.append(audio_tensor.shape[0])
		AUDIO = pad_sequence(AUDIO, padding_value=1.)
		# print('encoder', AUDIO.shape)
		# print(X_len)
		return AUDIO, X_len

class OverLastDim(nn.Module):
	def __init__(self, module):
		super().__init__()
		self.module = module

	def forward(self, x):
		*dims, input_size = x.size()

		reduced_dims = 1
		for dim in dims:
			reduced_dims *= dim

		x = x.view(reduced_dims, -1)
		x = self.module(x)
		x = x.view(*dims, -1)
		return x

class RNNWrapper(nn.Module):
	def __init__(self, input_size, hidden_size, rnn_type=nn.GRU,
				 bidirectional=True, batch_norm=True):
		super().__init__()
		if batch_norm:
			self.batch_norm = OverLastDim(nn.BatchNorm1d(input_size))
		self.bidirectional = bidirectional
		self.rnn = rnn_type(input_size=input_size,
							hidden_size=hidden_size,
							bidirectional=bidirectional,
							bias=False)

	def forward(self, x):
		if hasattr(self, 'batch_norm'):
			x = self.batch_norm(x)
		x, _ = self.rnn(x)
		if self.bidirectional:
			# TxNx(H*2) -> TxNxH by sum.
			seq_len, batch_size, _ = x.size()
			x = x.view(seq_len, batch_size, 2, -1) \
				 .sum(dim=2) \
				 .view(seq_len, batch_size, -1)
		return x

class Decoder(nn.Module):
	def __init__(self, in_features=80, n_hidden=MAX_LEN, out_features=VOCAB_LEN, rnn_layers=3, relu_clip=20.):
		super().__init__()
	
		# CONVOLUTIONAL layers
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels=1,
					  out_channels=32,
					  kernel_size=5,
					  stride=1,
					  padding='same'),
			nn.BatchNorm2d(32),
			nn.Hardtanh(0, relu_clip, inplace=True),
			nn.Conv2d(in_channels=32,
					  out_channels=32,
					  kernel_size=5,
					  stride=1,
					  padding='same'),
			nn.BatchNorm2d(32),
			nn.Hardtanh(0, relu_clip, inplace=True)
		)

		# RECURRENT layers
		rnn_in_size = 2560
		rnns = OrderedDict()
		for i in range(rnn_layers):
		  rnn = RNNWrapper(input_size=rnn_in_size,
							hidden_size=n_hidden,
							rnn_type=nn.GRU,
							bidirectional=True,
							batch_norm=i > 0)
		  rnns[str(i)] = rnn
		  rnn_in_size = n_hidden
		self.rnns = nn.Sequential(rnns)

		# FULLY CONNECTED layers
		fully_connected = nn.Sequential(
			nn.BatchNorm1d(n_hidden),
			nn.Linear(n_hidden, out_features, bias=False)
		)
		self.fc = OverLastDim(fully_connected)

	# for training
	def forward(self, X_in, X_len, Y_in, Y_len):
		"""
		Perform token prediction and compute loss over training set.

		Inputs:
		- X_in: A tensor of shape (seq_len, batch, in_features)
		  containing a mini-batch of audio features padded to seq_len.
		- Y_in: A tensor of shape (batch, max_seq_len)
		  containing a mini-batch of text targets padded to max_seq_len.
		  Each element in the target sequence is an index in the vocabulary. 
		  And the target index cannot be blank (index=0 in vocab).
		- Y_len: A tuple of shape (batch, ) containing the 
		  actual lengths of the targets (each <= max_seq_len).

		Returns:
		- loss: A PyTorch scalar containing the CTC loss for the mini-batch.
		"""
		# training logic
		# print('before conv', X_in.shape)
		X_in = X_in.permute(1, 2, 0)   # TxNxH -> NxHxT
		X_in.unsqueeze_(dim=1)      # NxHxT -> Nx1xHxT
		X_in = self.conv(X_in)
		# print('after conv', X_in.shape)

		N, H1, H2, T = X_in.size()
		x = X_in.view(N, H1*H2, T)
		x = x.permute(2, 0, 1)   # NxHxT -> TxNxH
		x = self.rnns(x.contiguous())
		# print('after rnns', x.shape)

		out = self.fc(x)
		logprobs = nn.functional.log_softmax(out, dim=2)
		# print('decoder', logprobs.shape)

		# compute CTC loss
		ctc_loss = nn.CTCLoss(zero_infinity=True)
		loss = ctc_loss(logprobs, Y_in, X_len, Y_len)
		return loss 

	# for inference
	def predict(self, X_in):
		"""
		Perform token prediction over validation/test set.

		Inputs:
		- X_in: A LongTensor of shape (batch_size, )
		  containing a mini-batch of input for inference.

		Returns:
		- text: A Tensor of shape (batch_size, MAX_LEN)
		  containing text output.
		"""
		# inference logic here
		X_in = X_in.permute(1, 2, 0)   # TxNxH -> NxHxT
		X_in.unsqueeze_(dim=1)      # NxHxT -> Nx1xHxT
		X_in = self.conv(X_in)

		N, H1, H2, T = X_in.size()
		x = X_in.view(N, H1*H2, T)
		x = x.permute(2, 0, 1)   # NxHxT -> TxNxH
		x = self.rnns(x.contiguous())
		out = self.fc(x)
		logprobs = nn.functional.log_softmax(out, dim=2)
		_, max_indices = logprobs.float().max(2)

		batch_sentences = []
		for i, indices in enumerate(max_indices.t()):
			no_dups, prev = [], None
			for index in indices:
				if prev is None or index != prev:
					no_dups.append(index.item())
					prev = index

			symbols = [vocab[s] for s in no_dups]

			no_blanks = [s for s in symbols if (s!=vocab[0] and s!=vocab[1])]
			batch_sentences.append(''.join(no_blanks))
		return batch_sentences

class CustomDataset(Dataset):
	def __init__(self, listOfFiles):
		self.X = listOfFiles
	
	def __len__(self):
		return len(self.X)
   
	def __getitem__(self, index):
		return self.X[index]

def delete_file_from_path(path):
	try:
		os.remove(path)
	except OSError as e:
		print("Error: %s : %s" % (path, e.strerror))

# execution starts here
if __name__ == "__main__":
	if len(sys.argv) <= 1:
		print("One argument required: python3 transcription.py [youtube_url]")
		sys.exit()
	youtube_url = sys.argv[1]

	# Get audio
	download_dir = './'
	yt = YouTube(youtube_url)
	ys = yt.streams.filter(only_audio=True, file_extension='mp4')
	print('Downloading : ', yt.title)
	ys[0].download(filename=yt.title+'.mp4')
	subprocess.call(['ffmpeg', \
		'-i', yt.title+'.mp4', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', \
		os.path.join(download_dir, yt.title+'.wav')])
	delete_file_from_path(yt.title+'.mp4')

	# Prepare audio
	myaudio = AudioSegment.from_wav(os.path.join(download_dir, yt.title+'.wav')) 
	chunk_length_ms = 8000 # pydub calculates in millisec 
	chunks = make_chunks(myaudio,chunk_length_ms) #Make chunks of one sec 
	for i, chunk in enumerate(chunks): 
		chunk_name = './chunked' + "_{0}.wav".format(i) 
		print ("exporting", chunk_name) 
		chunk.export(chunk_name, format="wav")
	delete_file_from_path(os.path.join(download_dir, yt.title+'.wav'))
	audio_list = [x for x  in os.listdir('./') if x.endswith(".wav")] 

	# Setup model
	enc = Encoder()
	dec = Decoder()

	checkpoint = None
	try:
	  checkpoint = torch.load(MODEL_CHECKPOINT, map_location=torch.device('cpu'))
	except Exception as e:
	  print(e)
	  pass

	if checkpoint is None or not bool(checkpoint):
		print('Model not found')
		sys.exit()

	if 'model_state_dict' in checkpoint:
		dec.load_state_dict(checkpoint['model_state_dict'])
	else:
		print('Model weights not found')
		sys.exit()

	# make inference
	dset = CustomDataset(audio_list)
	data_loader = DataLoader(dset, batch_size = batch_size, num_workers = workers)

	dec.eval()
	print ("transcribing chunks...")
	with open(yt.title+".txt", "a") as myfile:
		for idx, X_batch in enumerate(data_loader):
			# forward propagation
			X_in, X_len = enc(X_batch)
			Y_pred = dec.predict(X_in)
			for txt in Y_pred:
				myfile.write(txt)
	for file in audio_list:
		delete_file_from_path(file)
	print('Finished transcription...')