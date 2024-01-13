from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from ChatData import ChatData, MAX_LENGTH, START_STRING_TOKEN, END_STRING_TOKEN, BOT_TOKEN, PAD_TOKEN
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch

DATA_DIR = 'texts_qa'
EPOCHS = 100
SAVED_MODEL_NAME = 'gpt2-custom'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# Add the special tokens
tokenizer.add_special_tokens({
	'pad_token': PAD_TOKEN,
	'bos_token': START_STRING_TOKEN,
	'eos_token': END_STRING_TOKEN
})
tokenizer.add_tokens([BOT_TOKEN])

# Define your own GPT2LMHeadModel configuration
model_config = GPT2Config(
	vocab_size=tokenizer.vocab_size,
	n_positions=MAX_LENGTH,  # Adjust the maximum position length if necessary (gpt2 has 1024)
	n_embd=MAX_LENGTH*4,  # Adjust the embedding dimension (gpt2 has 768)
	n_layer=16,  # Adjust the number of layers (gpt2 has 12)
	n_head=16,  # Adjust the number of attention heads (gpt2 has 12)
	intermediate_size=1024,  # Adjust the intermediate size (gpt2 has 3072)
)
# Create model
model = GPT2LMHeadModel(config=model_config)
# reset model tokenizers since we added some custom tokens
model.resize_token_embeddings(len(tokenizer))
model = model.to(DEVICE)

# Tokenize data from the directory
chat_data = ChatData(DATA_DIR, tokenizer)
chat_data = DataLoader(chat_data, batch_size=8)

model.train()
optim = Adam(model.parameters(), lr=1e-4)

def infer(text_input):
	text_input = f'{START_STRING_TOKEN} {text_input.lower()} {BOT_TOKEN}'
	text_input = tokenizer(text_input, return_tensors='pt')
	X = text_input['input_ids'].to(DEVICE)
	a = text_input['attention_mask'].to(DEVICE)
	output = model.generate(X, attention_mask=a, max_length=MAX_LENGTH)
	return tokenizer.decode(output[0])

def train(chat_data, model, optim):
	for i in range(EPOCHS):
		for X, a in chat_data:
			X = X.to(DEVICE)
			a = a.to(DEVICE)
			optim.zero_grad()
			loss = model(X, attention_mask=a, labels=X).loss
			loss.backward()
			optim.step()
		if i % 5 == 0:
			test_sentence = 'What were your childhood aspirations for your future career?'
			print(f'Sentence: {infer(test_sentence)}')
			test_sentence = 'hindi kita maintindihan'
			print(f'Sentence: {infer(test_sentence)}\n')
		print(f'Epoch: {i+1}\tLoss: {loss.item()}')
	#torch.save(model.state_dict(), f'{SAVED_MODEL_NAME}.pt')
	# Save the fine-tuned model
	model.save_pretrained(SAVED_MODEL_NAME)
	tokenizer.save_pretrained(SAVED_MODEL_NAME)

train(chat_data, model, optim)
print(infer('mahal kita'))