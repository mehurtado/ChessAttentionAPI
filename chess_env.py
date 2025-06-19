# chess_env.py (Corrected for flexible board encoding)

import chess
import torch
import numpy as np
from typing import Tuple, List, Optional, Dict

import config

class ChessEnv:
    def __init__(self, board: Optional[chess.Board] = None):
        self.board = board if board is not None else chess.Board()
        # ... (the rest of the __init__ method is unchanged)
        self.policy_index_to_move_params, self.move_params_to_policy_index = self._create_move_map()

    def reset(self):
        self.board.reset()

    # --- THIS FUNCTION HAS BEEN MODIFIED ---
    def board_to_input_tensor(self, current_player_color: chess.Color, board: Optional[chess.Board] = None) -> torch.Tensor:
        """
        Converts a given board state to a 19x8x8 tensor.
        The perspective is always from the current_player_color.
        If no board is provided, it uses the instance's internal board.
        """
        # Use the provided board, or fall back to the instance's board
        target_board = board if board is not None else self.board

        tensor = torch.zeros((config.INPUT_CHANNELS, config.BOARD_SIZE, config.BOARD_SIZE), dtype=torch.float32)
        player = current_player_color
        
        piece_to_channel = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }

        for sq in chess.SQUARES:
            piece = target_board.piece_at(sq)
            if piece:
                rank, file = chess.square_rank(sq), chess.square_file(sq)
                if player == chess.BLACK:
                    rank = 7 - rank
                
                channel_offset = 0 if piece.color == player else 6
                channel = piece_to_channel[piece.piece_type] + channel_offset
                tensor[channel, rank, file] = 1

        if target_board.has_kingside_castling_rights(player): tensor[12, :, :] = 1
        if target_board.has_queenside_castling_rights(player): tensor[13, :, :] = 1
        if target_board.has_kingside_castling_rights(not player): tensor[14, :, :] = 1
        if target_board.has_queenside_castling_rights(not player): tensor[15, :, :] = 1

        tensor[16, :, :] = 1.0

        if target_board.ep_square is not None:
            rank, file = chess.square_rank(target_board.ep_square), chess.square_file(target_board.ep_square)
            if player == chess.BLACK: rank = 7 - rank
            tensor[17, rank, file] = 1
        
        tensor[18, :, :] = float(target_board.fullmove_number) / 100.0

        return tensor

    # --- All other functions (_create_move_map, action_to_policy_index, etc.) remain unchanged ---
    def _create_move_map(self) -> Tuple[List[Optional[Tuple[int, int, Optional[int]]]], Dict[Tuple[int, int, Optional[int]], int]]:
        policy_idx_to_move_params: List[Optional[Tuple[int, int, Optional[int]]]] = [None] * config.POLICY_OUTPUT_SIZE
        move_params_to_policy_idx: Dict[Tuple[int, int, Optional[int]], int] = {}
        queen_deltas = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
        knight_deltas_standard = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
        underpromotion_details = []
        for prom_piece in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
            underpromotion_details.append((-1, 1, prom_piece)); underpromotion_details.append((0, 1, prom_piece)); underpromotion_details.append((1, 1, prom_piece))
        action_plane_idx = 0
        for dr, df in queen_deltas:
            for dist in range(1, 8):
                plane_id = action_plane_idx
                for from_sq_wp in chess.SQUARES:
                    r_from, f_from = chess.square_rank(from_sq_wp), chess.square_file(from_sq_wp)
                    r_to, f_to = r_from + df * dist, f_from + dr * dist
                    if 0 <= r_to < 8 and 0 <= f_to < 8:
                        to_sq_wp = chess.square(f_to, r_to)
                        policy_idx = from_sq_wp * 73 + plane_id
                        if policy_idx < config.POLICY_OUTPUT_SIZE:
                            move_params = (from_sq_wp, to_sq_wp, None)
                            policy_idx_to_move_params[policy_idx] = move_params
                            move_params_to_policy_idx[move_params] = policy_idx
                action_plane_idx += 1
        for dr, df in knight_deltas_standard:
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
        for df, dr_prom, prom_piece_type in underpromotion_details:
            plane_id = action_plane_idx
            for to_f in range(8):
                to_sq_wp = chess.square(to_f, 7)
                from_f, from_r = to_f - df, 7 - dr_prom
                if 0 <= from_f < 8 and from_r == 6:
                    from_sq_wp = chess.square(from_f, from_r)
                    policy_idx = from_sq_wp * 73 + plane_id
                    if policy_idx < config.POLICY_OUTPUT_SIZE:
                        move_params = (from_sq_wp, to_sq_wp, prom_piece_type)
                        policy_idx_to_move_params[policy_idx] = move_params
                        if move_params not in move_params_to_policy_idx:
                             move_params_to_policy_idx[move_params] = policy_idx
            action_plane_idx +=1
        return policy_idx_to_move_params, move_params_to_policy_idx

    def action_to_policy_index(self, move: chess.Move, board: chess.Board) -> Optional[int]:
        from_sq_board, to_sq_board, promotion_board = move.from_square, move.to_square, move.promotion
        if board.turn == chess.WHITE:
            from_sq_wp, to_sq_wp, promotion_wp = from_sq_board, to_sq_board, promotion_board
        else:
            from_sq_wp, to_sq_wp, promotion_wp = chess.square_mirror(from_sq_board), chess.square_mirror(to_sq_board), promotion_board
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
        if not (0 <= index < config.POLICY_OUTPUT_SIZE): return None
        move_params_wp = self.policy_index_to_move_params[index]
        if move_params_wp is None: return None
        from_sq_wp, to_sq_wp, promotion_wp = move_params_wp
        if board.turn == chess.WHITE:
            from_sq_board, to_sq_board, promotion_board = from_sq_wp, to_sq_wp, promotion_wp
        else:
            from_sq_board, to_sq_board, promotion_board = chess.square_mirror(from_sq_wp), chess.square_mirror(to_sq_wp), promotion_wp
        potential_move = chess.Move(from_sq_board, to_sq_board, promotion=promotion_board)
        if potential_move in board.legal_moves:
            return potential_move
        else:
            piece_on_from_sq = board.piece_at(from_sq_board)
            if promotion_board is None and piece_on_from_sq is not None and piece_on_from_sq.piece_type == chess.PAWN:
                 if (board.turn == chess.WHITE and chess.square_rank(from_sq_board) == 6 and chess.square_rank(to_sq_board) == 7) or \
                    (board.turn == chess.BLACK and chess.square_rank(from_sq_board) == 1 and chess.square_rank(to_sq_board) == 0):
                    queen_prom_move = chess.Move(from_sq_board, to_sq_board, promotion=chess.QUEEN)
                    if queen_prom_move in board.legal_moves:
                        return queen_prom_move
            return None
    
    def get_legal_moves(self, board: chess.Board) -> List[chess.Move]:
        return list(board.legal_moves)

    def make_move(self, move: chess.Move):
        self.board.push(move)

    def is_game_over(self, board: chess.Board) -> bool:
        return board.is_game_over()

    def get_game_outcome(self, board: chess.Board) -> Optional[float]:
        if not board.is_game_over(): return None
        outcome = board.outcome();
        if outcome is None: return None
        if outcome.winner == chess.WHITE: return 1.0 if board.turn == chess.WHITE else -1.0
        elif outcome.winner == chess.BLACK: return -1.0 if board.turn == chess.WHITE else 1.0
        else: return 0.0

    def get_current_player_color(self, board: chess.Board) -> chess.Color:
        return board.turn