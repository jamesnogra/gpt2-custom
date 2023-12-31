from flask import Flask, request, Response, render_template
from flask_cors import CORS
import os

from testmodel import infer

GPT2_USERNAME = os.environ.get('GPT2_USERNAME')
GPT2_PASSWORD = os.environ.get('GPT2_PASSWORD')

app = Flask(__name__)
CORS(app)

def validate_basic_auth(request):
    auth = request.authorization
    if auth and auth.username == GPT2_USERNAME and auth.password == GPT2_PASSWORD:
        return True
    return False

@app.route('/chat', methods=['POST'])
def chat():
	if not validate_basic_auth(request):
		return Response('Invalid GPT2 API credentials', status=401)
	message = request.form.get('message')
	return infer(message)

if __name__ == '__main__':
	app.run(debug=True, port='8081', host='0.0.0.0', use_reloader=True)