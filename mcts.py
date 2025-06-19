# mcts.py (Optimized with Caching)

import chess
import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Optional, Tuple, List

import config
from chess_env import ChessEnv
from model import AttentionChessNet

class Node:
    def __init__(self, parent: Optional['Node'], prior_probability: float, board_state_at_node: chess.Board, player_to_move: chess.Color):
        self.parent: Optional['Node'] = parent
        self.children: Dict[chess.Move, 'Node'] = {}
        
        self.visit_count: int = 0
        self.total_action_value: float = 0.0
        self.prior_probability: float = prior_probability
        
        self.board_state_at_node: chess.Board = board_state_at_node.copy()
        self.is_expanded: bool = False
        self.player_to_move: chess.Color = player_to_move

    def Q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_action_value / self.visit_count

    def U_value(self, C_puct: float) -> float:
        if self.parent is None:
            total_parent_visits = self.visit_count
        else:
            total_parent_visits = self.parent.visit_count
            
        return C_puct * self.prior_probability * (math.sqrt(total_parent_visits) / (1 + self.visit_count))

    def select_child(self, C_puct: float) -> Tuple[chess.Move, 'Node']:
        best_score = -float('inf')
        best_action = None
        best_child_node = None

        for action, child_node in self.children.items():
            score = -child_node.Q_value() + child_node.U_value(C_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child_node = child_node
        
        if best_action is None or best_child_node is None:
            raise RuntimeError("select_child called on a node with no children and it's not terminal/expanded.")

        return best_action, best_child_node

class MCTS:
    def __init__(self, chess_env: ChessEnv, neural_net: AttentionChessNet, device: torch.device):
        self.chess_env = chess_env
        self.neural_net = neural_net
        self.device = device
        self.neural_net.to(self.device)
        # --- CACHE ---
        # Add a cache to store evaluations for the current search
        self.node_cache: Dict[str, Tuple[np.ndarray, float]] = {}

    def _get_policy_and_value(self, board: chess.Board, player_color: chess.Color) -> Tuple[np.ndarray, float]:
        """
        Gets policy and value from the neural network, using a cache to avoid re-evaluating nodes.
        """
        # --- CACHE LOOKUP ---
        # Use the board's FEN string as a unique key for the cache
        fen_key = board.fen()
        if fen_key in self.node_cache:
            return self.node_cache[fen_key]

        self.neural_net.eval()
        board_tensor = self.chess_env.board_to_input_tensor(player_color, board) # Pass board to encoder
        board_tensor = board_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value_tensor = self.neural_net(board_tensor)
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
             return np.array([]), value_tensor.item()

        masked_policy_logits = torch.full_like(policy_logits, -float('inf'))
        for move in legal_moves:
            idx = self.chess_env.action_to_policy_index(move, board)
            if idx is not None and 0 <= idx < config.POLICY_OUTPUT_SIZE:
                 masked_policy_logits[0, idx] = policy_logits[0, idx]

        policy_probs_tensor = F.softmax(masked_policy_logits, dim=1)
        policy_probs = policy_probs_tensor.squeeze(0).cpu().numpy()
        
        value = value_tensor.item()

        # --- CACHE STORE ---
        self.node_cache[fen_key] = (policy_probs, value)
        return policy_probs, value

    def run_simulations(self,
                        root_board_state: chess.Board,
                        num_simulations: int,
                        C_puct: float,
                        dirichlet_alpha: float = 0.0,
                        dirichlet_epsilon: float = 0.0
                       ) -> Tuple[Optional[chess.Move], np.ndarray, 'Node']:
        """
        Runs MCTS simulations from the root_board_state.
        """
        # --- CACHE CLEAR ---
        # Clear the cache at the start of each new move calculation
        self.node_cache.clear()

        root_player_color = root_board_state.turn
        root_node = Node(parent=None, prior_probability=1.0, board_state_at_node=root_board_state, player_to_move=root_player_color)

        if not root_node.board_state_at_node.is_game_over():
            policy_probs, _ = self._get_policy_and_value(root_node.board_state_at_node, root_player_color)
            self._expand_node(root_node, policy_probs, apply_dirichlet_noise=False) # No noise for inference
        
        for _ in range(num_simulations):
            current_node = root_node
            current_board = root_node.board_state_at_node.copy()
            
            path = [current_node]
            while current_node.is_expanded and not current_board.is_game_over():
                action, current_node = current_node.select_child(C_puct)
                current_board.push(action)
                path.append(current_node)

            value = 0.0
            if not current_board.is_game_over():
                if not current_node.is_expanded:
                    policy_probs_leaf, value_leaf = self._get_policy_and_value(current_board, current_board.turn)
                    self._expand_node(current_node, policy_probs_leaf)
                    value = value_leaf
            else:
                outcome = current_board.outcome(claim_draw=True)
                if outcome is not None:
                    player_at_leaf = current_board.turn 
                    if outcome.winner == chess.WHITE:
                        value = 1.0 if player_at_leaf == chess.WHITE else -1.0
                    elif outcome.winner == chess.BLACK:
                        value = -1.0 if player_at_leaf == chess.WHITE else 1.0
                    else:
                        value = 0.0
            
            for node_in_path in reversed(path):
                node_in_path.visit_count += 1
                node_in_path.total_action_value += value
                value = -value

        search_policy_vector = np.zeros(config.POLICY_OUTPUT_SIZE, dtype=np.float32)
        chosen_move = None
        if root_node.children:
            total_child_visits = sum(child.visit_count for child in root_node.children.values())
            best_move_visits = -1

            if total_child_visits > 0:
                for move, child_node in root_node.children.items():
                    policy_idx = self.chess_env.action_to_policy_index(move, root_board_state)
                    if policy_idx is not None and 0 <= policy_idx < config.POLICY_OUTPUT_SIZE:
                        search_policy_vector[policy_idx] = child_node.visit_count / total_child_visits
                    
                    if child_node.visit_count > best_move_visits:
                        best_move_visits = child_node.visit_count
                        chosen_move = move
        
        return chosen_move, search_policy_vector, root_node

    # Renamed to get_best_move to match the call in your chess_bot.py, even though run_simulations now returns it.
    # This is for backward compatibility with the provided chess_bot.py structure.
    # A better refactor would be to change chess_bot.py to use the tuple from run_simulations directly.
    def get_best_move(self, board_state, num_simulations):
         # This function is now a wrapper.
         best_move, _, _ = self.run_simulations(board_state, num_simulations, config.C_PUCT)
         return best_move

    def _expand_node(self, node_to_expand: Node, policy_probs: np.ndarray, apply_dirichlet_noise: bool = False, dirichlet_alpha: float = 0.0, dirichlet_epsilon: float = 0.0):
        if node_to_expand.is_expanded: return

        board = node_to_expand.board_state_at_node
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            node_to_expand.is_expanded = True
            return

        for move in legal_moves:
            child_board = board.copy()
            child_board.push(move)
            
            idx = self.chess_env.action_to_policy_index(move, board)
            prior = policy_probs[idx] if idx is not None and 0 <= idx < len(policy_probs) else 0.0

            child_node = Node(
                parent=node_to_expand,
                prior_probability=prior,
                board_state_at_node=child_board,
                player_to_move=child_board.turn
            )
            node_to_expand.children[move] = child_node
        
        node_to_expand.is_expanded = True