from torch.utils.data import Dataset
import os, csv

START_STRING_TOKEN = '<startofstring>'
END_STRING_TOKEN = '<endofstring>'
BOT_TOKEN = '<bot>:'
PAD_TOKEN = '<pad>'
MAX_LENGTH = 128

class ChatData(Dataset):

	def __init__(self, training_data_dir, tokenizer):
		self.X = []
		for filename in os.listdir(training_data_dir):
			file_path = os.path.join(training_data_dir, filename)
			print(f'Loading file {file_path}')
			with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
				# Use the csv.reader to handle quoted strings
				csv_reader = csv.reader(file)
				next(csv_reader)  # Skip header row
				for row in csv_reader:
					# Ensure there are two elements in the row
					if len(row) == 2:
						question, answer = row
						self.X.append(f'{START_STRING_TOKEN}{question.lower()} {BOT_TOKEN} {answer.lower()}{END_STRING_TOKEN}')
		self.X_encoded = tokenizer(
			self.X,
			max_length=MAX_LENGTH,
			padding='max_length',
			truncation=True,
			return_tensors='pt'
		)
		self.input_ids = self.X_encoded['input_ids']
		self.attention_mask = self.X_encoded['attention_mask']
		print(f'First sample data: {self.X[0]}')

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return (self.input_ids[idx], self.attention_mask[idx])