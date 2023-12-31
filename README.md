# Training the Model
- Add more data at folder texts_qa
- The format of the CSV should be question,answer and includes this header
- Run the command `python train.py`

# Testing the Model
- Run the command `python test.py`
- Type `exit` if done testing

# Running on Ubuntu Server
- Run `pip install -r requirements.txt`
- To run, type `python app-gpt2.pt`
- To run in background run `nohup python app-gpt2.py > log.txt 2>&1 &`
- To stop the background app running, type `ps aux | grep app-gpt2.py` and `kill PID`