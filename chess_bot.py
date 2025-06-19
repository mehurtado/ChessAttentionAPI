# chess_bot.py
# This file acts as a high-level interface for the chess engine.

import torch
import chess
import config  # Import the central configuration
from model import AttentionChessNet
from mcts import MCTS
from chess_env import ChessEnv
from utils import get_device

class ChessBot:
    """
    A wrapper class that encapsulates the chess engine's logic,
    providing a clean interface to get the best move for a given board state.
    This class is intended to be used by a web backend (e.g., FastAPI).
    """
    def __init__(self, model_path: str):
        """
        Initializes the chess bot by loading the model and setting up configurations.

        Args:
            model_path (str): Path to the trained PyTorch model checkpoint (.pth file).
        """
        # Load configuration from your config.py file
        self.device = get_device()
        self.mcts_simulations = config.MCTS_SIMULATIONS_EVAL

        print(f"Initializing ChessBot on device: {self.device}")
        print(f"Using {self.mcts_simulations} MCTS simulations per move.")

        # Instantiate the neural network from your model.py
        self.model = AttentionChessNet(
            d_model=config.D_MODEL,
            n_heads=config.N_HEADS,
            num_encoder_layers=config.NUM_ENCODER_LAYERS,
            dim_feedforward_scale=config.DIM_FEEDFORWARD_SCALE,
            dropout_rate=config.DROPOUT_RATE
        ).to(self.device)

        # Load the trained model weights
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Successfully loaded model weights from {model_path}")

            # Quantize the model for faster CPU inference
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("Model has been quantized for CPU performance.")
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}. The bot will use random weights.")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}. The bot will use random weights.")

        # Set the model to evaluation mode (important for inference)
        self.model.eval()

        # Instantiate the chess environment and MCTS engine once
        self.chess_env = ChessEnv()
        self.mcts_instance = MCTS(chess_env=self.chess_env, neural_net=self.model, device=self.device)


    def get_move(self, fen: str) -> str:
        """
        Calculates and returns the best move for a given FEN position.

        Args:
            fen (str): The board state in Forsyth-Edwards Notation.

        Returns:
            str: The best move in UCI format (e.g., 'e2e4'), or an empty string if the game is over.
        """
        if not self._is_valid_fen(fen):
            raise ValueError("Invalid FEN string provided.")

        # Create a python-chess board object for the current state
        board = chess.Board(fen)

        if board.is_game_over():
            print("Game is over, no move to make.")
            return ""

        # Run the MCTS search using the existing instance
        # For evaluation/play, we don't use Dirichlet noise
        best_move, _, _ = self.mcts_instance.run_simulations(
            root_board_state=board,
            num_simulations=self.mcts_simulations,
            C_puct=config.C_PUCT,
            dirichlet_alpha=0.0, # No noise for inference
            dirichlet_epsilon=0.0  # No noise for inference
        )
        
        if best_move is None:
            # This is an edge case, but as a fallback, we can choose a random legal move.
            print("Warning: MCTS returned no best move. Falling back to a random legal move.")
            if board.legal_moves.count() > 0:
                return list(board.legal_moves)[0].uci()
            else:
                return "" # No legal moves

        return best_move.uci()

    def _is_valid_fen(self, fen: str) -> bool:
        """A simple helper to check for FEN validity using the python-chess library."""
        try:
            chess.Board(fen)
            return True
        except (ValueError, TypeError):
            return False

# This block allows you to test the ChessBot class directly by running `python chess_bot.py`
if __name__ == '__main__':
    # CORRECTED PATH: Point to the model inside your new API project structure
    MODEL_CHECKPOINT_PATH = "./trained_models/current_best_nn.pth"
    
    print("--- ChessBot Local Test ---")
    try:
        # Initialize the bot
        bot = ChessBot(model_path=MODEL_CHECKPOINT_PATH)

        # --- Test Case 1: Starting position ---
        start_fen = chess.STARTING_FEN
        print(f"\nTesting position (FEN): {start_fen}")
        move = bot.get_move(start_fen)
        print(f"Bot's move: {move}")
        # ---

        # --- Test Case 2: A mid-game position ---
        mid_game_fen = "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3" # Sicilian Defense
        print(f"\nTesting position (FEN): {mid_game_fen}")
        move = bot.get_move(mid_game_fen)
        print(f"Bot's move: {move}")
        # ---

    except Exception as e:
        print(f"\nAn error occurred during the test: {e}")