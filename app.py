# app.py
# This file creates the FastAPI web server that serves the chess bot's moves.

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import chess

# Import the ChessBot class from your chess_bot.py file
from chess_bot import ChessBot

# --- 1. App and Bot Initialization ---

app = FastAPI()

# The path to your trained model file within this project
MODEL_PATH = "./trained_models/current_best_nn.pth"

# Create a single, global instance of your ChessBot.
# This ensures the model is loaded into memory only once when the server starts.
bot = ChessBot(model_path=MODEL_PATH)

# --- 2. CORS (Cross-Origin Resource Sharing) Middleware ---

# This is crucial for allowing your GitHub Pages frontend to communicate
# with this backend server, which will be running on a different domain.
# For development, we can allow all origins ("*").
# For production, you should restrict this to your actual website's domain.
origins = [
    "https://mehurtado.github.io", # Your production frontend
    "http://localhost",            # For local testing
    "http://127.0.0.1",            # For local testing
    "null"                         # For local file testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 3. API Endpoints ---

@app.get("/")
def read_root():
    """A simple health check endpoint to confirm the server is running."""
    return {"status": "Chess Bot API is running."}


@app.websocket("/ws/move")
async def websocket_endpoint(websocket: WebSocket):
    """
    The main WebSocket endpoint for real-time chess games.
    It receives a FEN string from the client, gets the bot's move,
    and sends it back.
    """
    await websocket.accept()
    print("WebSocket connection established.")
    try:
        while True:
            # Wait for a FEN string from the frontend
            fen = await websocket.receive_text()
            print(f"Received FEN: {fen}")

            # Get the bot's move using the existing bot instance
            # This can take a few seconds depending on MCTS simulations
            bot_move_uci = bot.get_move(fen)

            if bot_move_uci:
                print(f"Sending move: {bot_move_uci}")
                # Send the calculated move back to the frontend
                await websocket.send_text(bot_move_uci)
            else:
                # Handle game over or no-move scenarios
                print("Game over or no valid moves.")
                await websocket.send_text(" endgame") # Send a special message
    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"An error occurred: {e}")
        await websocket.close(code=1011, reason=str(e))


# --- 4. Running the Server ---

# This block allows you to run the server directly with `python app.py`
# for easy local development.
if __name__ == "__main__":
    # Use uvicorn to run the FastAPI app
    # host="0.0.0.0" makes it accessible on your local network
    # port=8000 is the standard for local dev servers
    uvicorn.run(app, host="0.0.0.0", port=8000)