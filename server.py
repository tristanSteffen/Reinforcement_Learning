from flask import Flask, request, jsonify
import numpy as np
from agent import DQNAgent  # Changed import

app = Flask(__name__, static_folder="static", static_url_path="")

# Initialize DQNAgent and load trained model
agent = DQNAgent()
agent.load("policy_p2.pt")  # Use .pt extension for PyTorch models

@app.route("/")
def serve_index():
    return app.send_static_file("index.html")

@app.route("/get_ai_move", methods=["POST"])
def get_ai_move():
    data = request.get_json()
    board_list = data["board"]
    
    # Convert to numpy array and reshape
    board_np = np.array(board_list, dtype=int)
    board_np[board_np == 2] = -1  # Map O to -1
    
    # Get valid actions
    valid_actions = [i for i, val in enumerate(board_np) if val == 0]
    
    if not valid_actions:
        return jsonify({"action": -1})

    # Get agent's action
    agent.epsilon = 0
    action = agent.act(board_np, valid_actions=valid_actions)
    
    return jsonify({"action": int(action)})

if __name__ == "__main__":
    app.run(debug=True)