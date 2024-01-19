print('Loading GPT2 libraries')
import os, re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from ChatData import START_STRING_TOKEN, END_STRING_TOKEN, BOT_TOKEN, PAD_TOKEN, MAX_LENGTH

SAVED_MODEL_NAME = 'gpt2-custom'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the fine-tuned model and tokenizer
print('Loading GPT2 head model and tokenizer')
model = GPT2LMHeadModel.from_pretrained(SAVED_MODEL_NAME)
tokenizer = GPT2Tokenizer.from_pretrained(SAVED_MODEL_NAME)
model.to(DEVICE)

def infer(text_input):
	text_input = f'{START_STRING_TOKEN} {text_input.lower()} {BOT_TOKEN}'
	text_input = tokenizer(text_input, return_tensors='pt')
	X = text_input['input_ids'].to(DEVICE)
	a = text_input['attention_mask'].to(DEVICE)
	# Reset model's hidden state
	with torch.no_grad():
		_ = model.forward(X, attention_mask=a)
	output = model.generate(X, attention_mask=a, max_length=MAX_LENGTH, top_k=5, do_sample=True)
	output = tokenizer.decode(output[0]) + END_STRING_TOKEN # Append the end of string token to make sure there is always one
	# Construct the regular expression pattern using variables
	pattern = re.escape(BOT_TOKEN) + r'\s*(.*?)\s*' + re.escape(END_STRING_TOKEN)
	match = re.search(pattern, output)
	if match:
		final_response = match.group(1)
		final_response = final_response.replace(START_STRING_TOKEN, '')
		final_response = final_response.replace(END_STRING_TOKEN, '')
		final_response = final_response.replace(BOT_TOKEN, '')
		final_response = final_response.replace(PAD_TOKEN, '')
		return final_response
	else:
		return ''