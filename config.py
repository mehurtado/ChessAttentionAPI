# config.py
# Objective: Centralized configuration for hyperparameters and settings.

# --- General ---
PROJECT_NAME = "AttentionalAlphaZeroChess"
DEVICE = "cuda" # "cuda" or "cpu"
RANDOM_SEED = 42

# --- Chess Logic Wrapper (chess_env.py) ---
# Board State Representation
INPUT_CHANNELS = 19 # 6 white, 6 black, 4 castling, 1 side_to_move, 1 en_passant, 1 move_count
BOARD_SIZE = 8
POLICY_OUTPUT_SIZE = 4672 # 8x8x73 (Queen moves + Knight moves + Underpromotions)

# --- Neural Network (model.py) ---
# Model Architecture
D_MODEL = 256           # Embedding dimension (e.g., 256 or 384)
N_HEADS = 4             # Number of attention heads (e.g., 4 or 8)
NUM_ENCODER_LAYERS = 6  # Number of Transformer encoder blocks (e.g., 4-8)
DIM_FEEDFORWARD_SCALE = 4 # Factor for FFN dimension in Transformer blocks (e.g., 4 * d_model)
DROPOUT_RATE = 0.1

# --- MCTS (mcts.py) ---
C_PUCT = 1.5            # Exploration constant in PUCT (e.g., 1.0 - 2.5)
DIRICHLET_ALPHA = 0.3   # Alpha for Dirichlet noise (e.g., 0.3 for chess)
DIRICHLET_EPSILON = 0.25 # Epsilon for Dirichlet noise (e.g., 0.25)
MCTS_SIMULATIONS_SELF_PLAY = 800 # Number of MCTS simulations per move during self-play
MCTS_SIMULATIONS_EVAL = 200    # Number of MCTS simulations per move during evaluation

# --- Self-Play (self_play.py) ---
TEMPERATURE_MOVES = 30  # Number of moves to use temperature sampling (τ=1.0)
REPLAY_BUFFER_SIZE = 500_000 # Max size of the replay buffer
NUM_SELF_PLAY_GAMES_PER_ITERATION = 15 # Number of games to generate per AlphaZero iteration

# --- Training (train.py) ---
LEARNING_RATE = 3e-4
BATCH_SIZE = 256        # (e.g., 256-512, constrained by VRAM)
L2_REG_CONST = 1e-4
NUM_TRAINING_EPOCHS_PER_ITERATION = 5 # Number of epochs to train per AlphaZero iteration
# Or NUM_TRAINING_STEPS_PER_ITERATION = 1000
GRAD_CLIP_NORM = None # Optional: max norm for gradient clipping

# --- Evaluation (evaluate.py) ---
NUM_EVAL_GAMES = 20     # Number of games to play for evaluation
EVAL_WIN_RATE_THRESHOLD = 0.55 # Threshold to accept new model

# --- Main Loop (main.py) ---
TOTAL_AZ_ITERATIONS = 1000 # Total number of AlphaZero iterations

# --- Logging & Checkpointing ---
LOG_LEVEL = "INFO"
CHECKPOINT_DIR = "./checkpoints"
TENSORBOARD_LOG_DIR = "./runs" # For TensorBoard or WandB project name