from testmodel import infer

while True:
	user_input = input("Enter a question or message (type 'exit' to quit): ")
	if user_input.lower() == 'exit':
		print("Exiting the program.")
		break
	print(infer(user_input))