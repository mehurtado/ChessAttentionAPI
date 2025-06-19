# chess_env.py
# Phase 2: Core Modules Implementation
# Module 1: Chess Logic Wrapper
# Objective: Abstract chess rules and state representation.

import chess
import torch
import numpy as np
from typing import Tuple, List, Optional, Dict

import config # Assuming your config.py is in the same directory or accessible

class ChessEnv:
    """
    Wrapper class around python-chess for game state management,
    move conversion, and board representation.
    """
    def __init__(self, board: Optional[chess.Board] = None):
        self.board = board if board is not None else chess.Board()
        # Initialize mappings:
        # policy_index_to_move_params: List[Optional[Tuple[int, int, Optional[int]]]]
        # move_params_to_policy_index: Dict[Tuple[int, int, Optional[int]], int]
        self.policy_index_to_move_params, self.move_params_to_policy_index = self._create_move_map()

    def reset(self):
        """Resets the board to the starting position."""
        self.board.reset()

    def board_to_input_tensor(self, current_player_color: chess.Color) -> torch.Tensor:
        """
        Converts the current board state to a 19x8x8 tensor.
        The perspective is always from the current_player_color.
        """
        tensor = torch.zeros((config.INPUT_CHANNELS, config.BOARD_SIZE, config.BOARD_SIZE), dtype=torch.float32)
        player = current_player_color
        
        piece_to_channel = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }

        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece:
                rank, file = chess.square_rank(sq), chess.square_file(sq)
                if player == chess.BLACK: # Flip rank for Black's perspective
                    rank = 7 - rank
                
                channel_offset = 0 if piece.color == player else 6
                channel = piece_to_channel[piece.piece_type] + channel_offset
                tensor[channel, rank, file] = 1

        # Castling rights (from player's perspective)
        # Plane 12: Player's Kingside
        # Plane 13: Player's Queenside
        # Plane 14: Opponent's Kingside
        # Plane 15: Opponent's Queenside
        if self.board.has_kingside_castling_rights(player): tensor[12, :, :] = 1
        if self.board.has_queenside_castling_rights(player): tensor[13, :, :] = 1
        if self.board.has_kingside_castling_rights(not player): tensor[14, :, :] = 1
        if self.board.has_queenside_castling_rights(not player): tensor[15, :, :] = 1

        tensor[16, :, :] = 1.0 # Side to move is always current player

        if self.board.ep_square is not None:
            rank, file = chess.square_rank(self.board.ep_square), chess.square_file(self.board.ep_square)
            if player == chess.BLACK: rank = 7 - rank
            tensor[17, rank, file] = 1
        
        tensor[18, :, :] = float(self.board.fullmove_number) / 100.0 # Example: fullmove number
        # Or self.board.halfmove_clock / 100.0

        return tensor

    def _create_move_map(self) -> Tuple[List[Optional[Tuple[int, int, Optional[int]]]], Dict[Tuple[int, int, Optional[int]], int]]:
        """
        Creates mappings for AlphaZero-style action representation (8x8x73 = 4672).
        Stores moves from White's perspective (from_sq_wp, to_sq_wp, promotion_wp).
        """
        policy_idx_to_move_params: List[Optional[Tuple[int, int, Optional[int]]]] = [None] * config.POLICY_OUTPUT_SIZE
        move_params_to_policy_idx: Dict[Tuple[int, int, Optional[int]], int] = {}

        # Directions for queen-like moves (N, NE, E, SE, S, SW, W, NW)
        queen_deltas = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1)
        ]
        # Standard knight move deltas (file_delta, rank_delta)
        knight_deltas_standard = [
            (1, 2), (2, 1), (-1, 2), (-2, 1),
            (1, -2), (2, -1), (-1, -2), (-2, -1)
        ]

        # Pawn underpromotion directions (relative to White: capture left, forward, capture right)
        underpromotion_details = [] # (delta_file, delta_rank, piece_type)
        for prom_piece in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
            underpromotion_details.append((-1, 1, prom_piece)) # Capture left
            underpromotion_details.append((0, 1, prom_piece))  # Forward
            underpromotion_details.append((1, 1, prom_piece))  # Capture right

        action_plane_idx = 0

        # 1. Queen-like moves (56 planes)
        for dir_idx, (df, dr) in enumerate(queen_deltas):
            for dist in range(1, 8): # Max 7 squares
                plane_id = action_plane_idx 
                for from_sq_wp in chess.SQUARES: 
                    r_from, f_from = chess.square_rank(from_sq_wp), chess.square_file(from_sq_wp)
                    r_to, f_to = r_from + dr * dist, f_from + df * dist
                    if 0 <= r_to < 8 and 0 <= f_to < 8:
                        to_sq_wp = chess.square(f_to, r_to)
                        policy_idx = from_sq_wp * 73 + plane_id
                        if policy_idx < config.POLICY_OUTPUT_SIZE: 
                            move_params = (from_sq_wp, to_sq_wp, None)
                            policy_idx_to_move_params[policy_idx] = move_params
                            move_params_to_policy_idx[move_params] = policy_idx
                action_plane_idx += 1
        
        # 2. Knight moves (8 planes)
        for knight_move_idx, (df, dr) in enumerate(knight_deltas_standard): 
            plane_id = action_plane_idx
            for from_sq_wp in chess.SQUARES:
                r_from, f_from = chess.square_rank(from_sq_wp), chess.square_file(from_sq_wp)
                r_to, f_to = r_from + dr, f_from + df
                if 0 <= r_to < 8 and 0 <= f_to < 8:
                    to_sq_wp = chess.square(f_to, r_to)
                    policy_idx = from_sq_wp * 73 + plane_id
                    if policy_idx < config.POLICY_OUTPUT_SIZE:
                        move_params = (from_sq_wp, to_sq_wp, None)
                        policy_idx_to_move_params[policy_idx] = move_params
                        move_params_to_policy_idx[move_params] = policy_idx
            action_plane_idx +=1

        # 3. Pawn underpromotions (9 planes)
        for prom_idx, (df, dr_prom, prom_piece_type) in enumerate(underpromotion_details):
            plane_id = action_plane_idx
            for to_f in range(8): 
                to_sq_wp = chess.square(to_f, 7) 
                from_f = to_f - df
                from_r = 7 - dr_prom 
                
                if 0 <= from_f < 8 and from_r == 6: 
                    from_sq_wp = chess.square(from_f, from_r)
                    policy_idx = from_sq_wp * 73 + plane_id 
                    
                    if policy_idx < config.POLICY_OUTPUT_SIZE:
                        move_params = (from_sq_wp, to_sq_wp, prom_piece_type)
                        policy_idx_to_move_params[policy_idx] = move_params
                        if move_params not in move_params_to_policy_idx: 
                             move_params_to_policy_idx[move_params] = policy_idx
                        elif move_params_to_policy_idx[move_params] != policy_idx :
                            pass
            action_plane_idx +=1
        
        assert action_plane_idx == 73, f"Expected 73 action planes, got {action_plane_idx}"
        return policy_idx_to_move_params, move_params_to_policy_idx

    def action_to_policy_index(self, move: chess.Move, board: chess.Board) -> Optional[int]:
        """
        Converts a chess.Move object to an integer index (0-4671) for the policy output.
        Perspective of the move is transformed to White's perspective for map lookup.
        Returns None if the move cannot be mapped (should not happen for legal moves).
        """
        from_sq_board = move.from_square
        to_sq_board = move.to_square
        promotion_board = move.promotion

        if board.turn == chess.WHITE:
            from_sq_wp = from_sq_board
            to_sq_wp = to_sq_board
            promotion_wp = promotion_board
        else: 
            from_sq_wp = chess.square_mirror(from_sq_board)
            to_sq_wp = chess.square_mirror(to_sq_board)
            promotion_wp = promotion_board 

        move_params_wp = (from_sq_wp, to_sq_wp, promotion_wp)
        
        if move_params_wp in self.move_params_to_policy_index:
            return self.move_params_to_policy_index[move_params_wp]
        else:
            if promotion_wp == chess.QUEEN:
                move_params_no_prom = (from_sq_wp, to_sq_wp, None)
                if move_params_no_prom in self.move_params_to_policy_index:
                    return self.move_params_to_policy_index[move_params_no_prom]
            return None


    def policy_index_to_action(self, index: int, board: chess.Board) -> Optional[chess.Move]:
        """
        Converts an integer index (0-4671) back to a chess.Move object
        for the current board state. Returns None if the move derived from index is illegal.
        """
        if not (0 <= index < config.POLICY_OUTPUT_SIZE):
            return None

        move_params_wp = self.policy_index_to_move_params[index]
        if move_params_wp is None:
            return None

        from_sq_wp, to_sq_wp, promotion_wp = move_params_wp

        if board.turn == chess.WHITE:
            from_sq_board = from_sq_wp
            to_sq_board = to_sq_wp
            promotion_board = promotion_wp
        else: 
            from_sq_board = chess.square_mirror(from_sq_wp)
            to_sq_board = chess.square_mirror(to_sq_wp)
            promotion_board = promotion_wp 
        
        potential_move = chess.Move(from_sq_board, to_sq_board, promotion=promotion_board)

        if potential_move in board.legal_moves:
            return potential_move
        else:
            piece_on_from_sq = board.piece_at(from_sq_board)
            if promotion_board is None and \
               piece_on_from_sq is not None and \
               piece_on_from_sq.piece_type == chess.PAWN:
                 if (board.turn == chess.WHITE and chess.square_rank(from_sq_board) == 6 and chess.square_rank(to_sq_board) == 7) or \
                    (board.turn == chess.BLACK and chess.square_rank(from_sq_board) == 1 and chess.square_rank(to_sq_board) == 0):
                    queen_prom_move = chess.Move(from_sq_board, to_sq_board, promotion=chess.QUEEN)
                    if queen_prom_move in board.legal_moves:
                        return queen_prom_move
            return None

    def get_legal_moves(self) -> List[chess.Move]:
        """Returns a list of legal moves for the current board state."""
        return list(self.board.legal_moves)

    def make_move(self, move: chess.Move):
        """Makes the given move on the board."""
        self.board.push(move)

    def is_game_over(self) -> bool:
        """Checks if the game is over."""
        return self.board.is_game_over()

    def get_game_outcome(self) -> Optional[float]:
        """
        Returns the game outcome from the perspective of the current player to move.
        1 for win, -1 for loss, 0 for draw. None if game is not over.
        """
        if not self.board.is_game_over():
            return None
        
        outcome = self.board.outcome() 
        if outcome is None: return None 

        if outcome.winner == chess.WHITE:
            return 1.0 if self.board.turn == chess.WHITE else -1.0
        elif outcome.winner == chess.BLACK:
            return -1.0 if self.board.turn == chess.WHITE else 1.0
        else: # Draw
            return 0.0

    def get_current_player_color(self) -> chess.Color:
        """Returns the color of the player whose turn it is."""
        return self.board.turn

# --- Testing ---
if __name__ == '__main__':
    env = ChessEnv()
    print("Initial board:")
    print(env.board)

    print(f"\nSize of policy_index_to_move_params: {len(env.policy_index_to_move_params)}")
    print(f"Number of entries in move_params_to_policy_index: {len(env.move_params_to_policy_index)}")
    
    valid_indices = sum(1 for item in env.policy_index_to_move_params if item is not None)
    print(f"Number of valid move_params in policy_index_to_move_params: {valid_indices}")


    print("\nTesting board_to_input_tensor for White:")
    tensor_white_pov = env.board_to_input_tensor(chess.WHITE)
    print(f"Shape: {tensor_white_pov.shape}")

    initial_legal_moves = env.get_legal_moves()
    if initial_legal_moves:
        test_move = initial_legal_moves[0] 
        print(f"\nTesting move: {test_move.uci()} for player {env.board.turn}")
        
        policy_idx = env.action_to_policy_index(test_move, env.board)
        print(f"Move {test_move.uci()} -> Policy Index: {policy_idx}")

        if policy_idx is not None:
            retrieved_move_obj = env.policy_index_to_action(policy_idx, env.board)
            print(f"Policy Index {policy_idx} -> Move: {retrieved_move_obj.uci() if retrieved_move_obj else 'None or Illegal'}")
            if retrieved_move_obj:
                assert retrieved_move_obj == test_move, f"Move conversion mismatch: {test_move} vs {retrieved_move_obj}"
                print("Forward and backward conversion matches.")
            else:
                print("Could not retrieve move from policy index or it's illegal (unexpected for a legal starting move).")
        else:
            print(f"Could not map move {test_move.uci()} to policy index (unexpected).")

        print(f"\nMaking move: {test_move.uci()}")
        env.make_move(test_move)
        print(env.board)

        print("\nTesting board_to_input_tensor for Black (current player):")
        tensor_black_pov = env.board_to_input_tensor(chess.BLACK)
        print(f"Shape: {tensor_black_pov.shape}")

        blacks_legal_moves = env.get_legal_moves()
        if blacks_legal_moves:
            black_test_move = blacks_legal_moves[0]
            print(f"\nTesting Black's move: {black_test_move.uci()} for player {env.board.turn}")
            
            policy_idx_black = env.action_to_policy_index(black_test_move, env.board)
            print(f"Move {black_test_move.uci()} -> Policy Index: {policy_idx_black}")

            if policy_idx_black is not None:
                retrieved_black_move_obj = env.policy_index_to_action(policy_idx_black, env.board)
                print(f"Policy Index {policy_idx_black} -> Move: {retrieved_black_move_obj.uci() if retrieved_black_move_obj else 'None or Illegal'}")
                if retrieved_black_move_obj:
                    assert retrieved_black_move_obj == black_test_move, f"Black's move conversion mismatch: {black_test_move} vs {retrieved_black_move_obj}"
                    print("Black's move forward and backward conversion matches.")
                else:
                    print("Could not retrieve Black's move from policy index or it's illegal.")
            else:
                 print(f"Could not map Black's move {black_test_move.uci()} to policy index (unexpected).")

    print("\nSimulating a quick game for outcome test (Fool's Mate):")
    env.reset()
    try:
        env.make_move(env.board.parse_uci("f2f3"))
        env.make_move(env.board.parse_uci("e7e5"))
        env.make_move(env.board.parse_uci("g2g4"))
        env.make_move(env.board.parse_uci("d8h4")) 
        
        print(env.board)
        if env.is_game_over():
            print("Game is over.")
            outcome_val = env.get_game_outcome() 
            print(f"Outcome for player whose turn it is (White): {outcome_val}")
            assert outcome_val == -1.0
        else:
            print("Game is not over yet (something wrong with mate sequence).")

    except Exception as e:
        print(f"Error during game simulation: {e}")
        import traceback
        traceback.print_exc()

    print("\nChessEnv tests completed with AlphaZero move mappings.")
    print("Review Knight deltas and underpromotion logic if issues persist.")
    print("The number of valid_indices should ideally be close to 4672, accounting for moves that are always off-board.")