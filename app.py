from flask import Flask, request, jsonify 
from flask_cors import CORS
import openai
import os
from flask import render_template
import requests

app = Flask(__name__)
CORS(app)

# Set your OpenAI API key

@app.route('/')
def serve_ui():
    return render_template('index.html')

#@app.route('/chooseavatar', methods=['POST'])
#def chooseavatar():
 #   response = request.post("")
    #data = response.json()


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    print("User said:", user_message)

    if not user_message:
        return jsonify({'error': 'Empty Message'}), 400

    try:
        #response = openai.ChatCompletion.create(
         #   model="gpt-3.5-turbo",
          #  messages=[{"role": "user", "content": user_message}]
        #)
        
        
        #reply = response['choices'][0]['message']['content']
        #return jsonify({'reply': reply})
    #except Exception as e:
     #   print("OpenAI Error:", e)
     #   return jsonify({'error': str(e)}), 500

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": user_message, "stream": False}
        )
        data = response.json()
        return jsonify({'reply': data.get('response', 'No response from model.')})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
