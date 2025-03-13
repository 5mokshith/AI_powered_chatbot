from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from inference.inference import PolicyQASystem

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

qa_system = PolicyQASystem(
    data_path=r'C:\Users\moksh\classroom\chatbot_deepseek\industry_chatbot\data\knowledge_base\cleaned_augmented_qa_pairs.json'
)

@app.route('/api/ask', methods=['POST'])
def ask_question():
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    response = qa_system.get_answer(query)
    return jsonify({
        "question": query,
        "answer": response
    })

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)