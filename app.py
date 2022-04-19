from flask import Flask,request,jsonify
from chatting import ChatWithBot
import json
app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, Flask!"

@app.route('/chat',methods=['POST'])
def chatBot():
    chatInput = request.get_json()
    return jsonify(chatBotReply=ChatWithBot(chatInput["chatInput"]))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000,debug=True)