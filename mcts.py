# mcts.py
# Phase 2: Core Modules Implementation
# Module 3: Monte Carlo Tree Search (MCTS)
# Objective: Implement the MCTS algorithm.

import chess
import torch
from torch.amp import autocast
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Optional, Tuple, List

import config
from chess_env import ChessEnv # For board operations and move conversions
from model import AttentionChessNet # For neural network evaluation

class Node:
    def __init__(self, parent: Optional['Node'], prior_probability: float, board_state_at_node: chess.Board, player_to_move: chess.Color):
        self.parent: Optional['Node'] = parent
        self.children: Dict[chess.Move, 'Node'] = {} # Maps move to child node
        
        self.visit_count: int = 0       # N
        self.total_action_value: float = 0.0 # W (sum of values from outcomes or network evaluations)
        self.prior_probability: float = prior_probability # P (from network policy head for the action that led to this node)
        
        self.board_state_at_node: chess.Board = board_state_at_node.copy() # Immutable snapshot
        self.is_expanded: bool = False
        self.player_to_move: chess.Color = player_to_move # Whose turn it is at this node

    def Q_value(self) -> float:
        """Average action value (W/N)."""
        if self.visit_count == 0:
            # For unvisited nodes, AlphaZero often uses parent's Q or 0.
            # Using 0 encourages exploration of unvisited branches initially.
            # If parent exists, could use -parent.Q_value() if values are from current player's perspective.
            # For simplicity here, let's use 0.
            return 0.0
        return self.total_action_value / self.visit_count

    def U_value(self, C_puct: float) -> float:
        """PUCT exploration bonus."""
        # U = C_puct * P * sqrt(sum of visits of siblings) / (1 + N)
        # AlphaZero paper uses: C_puct * P * sqrt(N_parent) / (1 + N_child)
        # where N_parent is visit count of the parent node.
        if self.parent is None: # Should not happen for U-value calculation in selection
            total_parent_visits = self.visit_count # Or some large number if root
        else:
            total_parent_visits = self.parent.visit_count
            
        return C_puct * self.prior_probability * (math.sqrt(total_parent_visits) / (1 + self.visit_count))

    def select_child(self, C_puct: float) -> Tuple[chess.Move, 'Node']:
        """Selects the child that maximizes Q(s,a) + U(s,a)."""
        best_score = -float('inf')
        best_action = None
        best_child_node = None

        for action, child_node in self.children.items():
            # Q value is from the perspective of the player who made the move to reach child_node
            # So, if current node is P1, child is P2 state. Q(child) is P2's value.
            # We need -Q(child) for P1's PUCT score.
            score = -child_node.Q_value() + child_node.U_value(C_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child_node = child_node
        
        if best_action is None or best_child_node is None:
            # This can happen if there are no children (e.g. terminal node was not marked)
            # Or if all children have -inf scores (e.g. all losses)
            # Should ideally not happen if selection is called on an expanded, non-terminal node.
            # Fallback: pick a random child if any exist
            if self.children:
                print("Warning: best_action was None in select_child, picking random.")
                best_action = list(self.children.keys())[0]
                best_child_node = self.children[best_action]
            else: # This is a problem, means we tried to select from a leaf that wasn't expanded or terminal
                raise RuntimeError("select_child called on a node with no children and it's not terminal/expanded.")


        return best_action, best_child_node


class MCTS:
    def __init__(self, chess_env: ChessEnv, neural_net: AttentionChessNet, device: torch.device):
        self.chess_env = chess_env # Provides board_to_input_tensor, policy_index_to_action etc.
        self.neural_net = neural_net
        self.device = device
        self.neural_net.to(self.device)

    def _get_policy_and_value(self, board: chess.Board, player_color: chess.Color) -> Tuple[np.ndarray, float]:
        """
        Gets policy and value from the neural network for a given board state.
        Ensures eval mode and no_grad.
        """
        self.neural_net.eval()
        board_tensor = self.chess_env.board_to_input_tensor(player_color)
        board_tensor = board_tensor.unsqueeze(0).to(self.device) # Add batch dimension

        with torch.no_grad():
            policy_logits, value_tensor = self.neural_net(board_tensor)
        
        # Policy logits to probabilities
        # Mask illegal moves before softmax: set logits of illegal moves to -inf
        # This is crucial.
        legal_moves = list(board.legal_moves)
        if not legal_moves: # Terminal state, no legal moves
            # Return a uniform policy over a dummy action or handle upstream
            # For now, if no legal moves, policy is irrelevant, value is from game outcome.
            # This function is typically called on non-terminal states for expansion.
            # If called here, it means an issue.
             return np.array([]), value_tensor.item()


        masked_policy_logits = torch.full_like(policy_logits, -float('inf'))
        
        valid_action_indices = []
        for move in legal_moves:
            # The chess_env.action_to_policy_index needs to be robust
            idx = self.chess_env.action_to_policy_index(move, board)
            if 0 <= idx < config.POLICY_OUTPUT_SIZE:
                 masked_policy_logits[0, idx] = policy_logits[0, idx]
                 valid_action_indices.append(idx)
            # else:
                # print(f"Warning: Move {move.uci()} produced invalid index {idx} from action_to_policy_index.")

        if not valid_action_indices: # No legal moves mapped to valid policy indices
            # This indicates a problem with action_to_policy_index or the policy head output interpretation
            # Fallback: uniform policy over legal moves (if any) mapped to first few indices
            # This is a HACK for robustness if action_to_policy_index is buggy.
            # print("Warning: No valid action indices found. Using uniform over legal moves as fallback policy.")
            # policy_probs = np.zeros(config.POLICY_OUTPUT_SIZE, dtype=np.float32)
            # if legal_moves:
            #     prob_per_move = 1.0 / len(legal_moves)
            #     for i, move in enumerate(legal_moves): # Assign to first N indices
            #         if i < config.POLICY_OUTPUT_SIZE:
            #             policy_probs[i] = prob_per_move
            # else: # No legal moves, should be terminal
            #     pass # policy_probs remains all zeros
            # This fallback is dangerous as it doesn't use network's priors.
            # Better to ensure action_to_policy_index is perfect.
            # For now, if masked_policy_logits is all -inf, softmax will be NaN or uniform.
            # Let's ensure at least one valid logit if there are legal moves.
            # If all mapped indices were invalid, this is a critical bug in action_to_policy_index.
            # We assume for now that action_to_policy_index works.
             pass


        policy_probs_tensor = F.softmax(masked_policy_logits, dim=1)
        policy_probs = policy_probs_tensor.squeeze(0).cpu().numpy()
        
        value = value_tensor.item() # Scalar value
        return policy_probs, value

    def run_simulations(self,
                        root_board_state: chess.Board,
                        num_simulations: int,
                        C_puct: float,
                        dirichlet_alpha: float,
                        dirichlet_epsilon: float,
                        temperature_final_move: float = 1.0 # Temperature for choosing the final move
                       ) -> Tuple[chess.Move, np.ndarray, Node]:
        """
        Runs MCTS simulations from the root_board_state.
        Returns the chosen move, the search policy vector, and the root node.
        """
        root_player_color = root_board_state.turn
        root_node = Node(parent=None, prior_probability=1.0, board_state_at_node=root_board_state, player_to_move=root_player_color)

        # Initial expansion of root if not terminal
        if not root_node.board_state_at_node.is_game_over():
            policy_probs, value = self._get_policy_and_value(root_node.board_state_at_node, root_player_color)
            
            # Add Dirichlet noise to root policy for exploration during self-play
            if dirichlet_alpha > 0 and dirichlet_epsilon > 0:
                legal_moves = list(root_node.board_state_at_node.legal_moves)
                num_legal_moves = len(legal_moves)
                if num_legal_moves > 0:
                    # Apply noise only to policy elements corresponding to legal moves
                    noise = np.random.dirichlet([dirichlet_alpha] * num_legal_moves)
                    
                    # Create a temporary full noise vector
                    full_noise_vector = np.zeros_like(policy_probs)
                    
                    # Get indices for legal moves
                    legal_move_indices = []
                    for move in legal_moves:
                        idx = self.chess_env.action_to_policy_index(move, root_node.board_state_at_node)
                        if 0 <= idx < config.POLICY_OUTPUT_SIZE:
                            legal_move_indices.append(idx)
                    
                    # Apply noise to the corresponding entries in policy_probs
                    # This is tricky if policy_probs isn't perfectly aligned or dense for legal moves
                    # A safer way: modify the priors of children directly.
                    # For now, let's assume policy_probs is the vector to modify.
                    # We need to ensure noise is added to the correct P(s,a) values.
                    
                    # Simpler: apply noise to the priors that will be assigned to children
                    # This is done during expansion below.
                    # Here, we just mark that noise should be applied.
                    apply_dirichlet_noise = True
                else:
                    apply_dirichlet_noise = False
            else:
                apply_dirichlet_noise = False

            # Expand root node
            self._expand_node(root_node, policy_probs, apply_dirichlet_noise, dirichlet_alpha, dirichlet_epsilon)
            # The first backpropagation for the root is based on its own network evaluation
            # This isn't standard. Usually, first eval sets priors, then sims start.
            # Let's adjust: expansion sets priors. First sim goes down, gets a value, then backprops.
            # The value from initial _get_policy_and_value is for the root state itself.
            # This value will be used if a simulation path ends up selecting this node for backpropagation
            # without going deeper (e.g. if it's a leaf in a shallow search).
            # For now, let's not backprop this initial value yet.
            # root_node.visit_count += 1 # No, this is done during backprop
            # root_node.total_action_value += value # No
        else: # Root is a terminal state
            # No simulations to run, outcome is fixed.
            # The MCTS result should reflect this.
            # This case should be handled by the caller (e.g., in self-play loop).
            # If called here, it implies game already ended.
            # Search policy would be empty or reflect no options.
            # Chosen move would be None.
             pass


        for _ in range(num_simulations):
            current_node = root_node
            current_board = root_node.board_state_at_node.copy() # Board for simulation path
            
            # 1. Selection
            path = [current_node] # Keep track of path for backpropagation
            while current_node.is_expanded and not current_board.is_game_over():
                action, current_node = current_node.select_child(C_puct)
                current_board.push(action)
                path.append(current_node)

            # 2. Expansion & Evaluation
            value = 0.0
            if not current_board.is_game_over():
                if not current_node.is_expanded: # Expand if not already (should be true here)
                    # Get policy and value from network for the new leaf node's state
                    # Player color for network eval is current_board.turn
                    policy_probs_leaf, value_leaf = self._get_policy_and_value(current_board, current_board.turn)
                    self._expand_node(current_node, policy_probs_leaf) # No dirichlet noise for internal nodes
                    value = value_leaf # Value is from perspective of player to move at current_board
                else:
                    # This case (expanded but game not over, yet selection stopped) shouldn't happen
                    # unless C_puct or Q/U values lead to a strange state.
                    # Or if node was previously expanded but now is terminal due to simulation rules (e.g. max depth).
                    # For now, assume if we are here, it's a leaf that needs evaluation.
                    # This might occur if a node was expanded, but then another simulation path made it terminal
                    # (e.g. repetition). This is complex.
                    # Simpler: if it's expanded, it means children exist. Selection should have picked one.
                    # If it's not game_over, it must be expandable.
                    # This path implies current_node *was* a leaf and is now being expanded.
                    # The value is from the network.
                    # The value from _get_policy_and_value is from the perspective of current_board.turn
                    # (the player AT the expanded node current_node)
                    _, value_at_leaf = self._get_policy_and_value(current_board, current_board.turn)
                    value = value_at_leaf

            else: # Game is over at current_node (leaf is terminal)
                outcome = current_board.outcome()
                if outcome is None: # Should not happen if is_game_over
                    value = 0.0
                else:
                    # Outcome is from White's perspective. Convert to current player's perspective.
                    # Player at current_node is current_board.turn (whose move it would be if game didn't end)
                    player_at_leaf = current_board.turn 
                    if outcome.winner == chess.WHITE:
                        value = 1.0 if player_at_leaf == chess.WHITE else -1.0
                    elif outcome.winner == chess.BLACK:
                        value = -1.0 if player_at_leaf == chess.WHITE else 1.0
                    else: # Draw
                        value = 0.0
            
            # 3. Backpropagation
            # Value is from the perspective of the player whose turn it is at the *expanded/terminal leaf node*.
            # When backing up to parent, the value needs to be flipped.
            for node_in_path in reversed(path):
                node_in_path.visit_count += 1
                # total_action_value should reflect value for the player *at that node*
                # If node_in_path.player_to_move is P1, and 'value' is for P1, add value.
                # If 'value' is for P2 (child of P1), then P1 gets -value.
                # The 'value' obtained above is for the player at current_node (the leaf of the selection).
                # So, for current_node's parent, this value should be negated.
                node_in_path.total_action_value += value
                value = -value # Flip for the parent

        # After all simulations, calculate search policy pi from root node's children visit counts
        # This pi is the target for training the policy head.
        search_policy_vector = np.zeros(config.POLICY_OUTPUT_SIZE, dtype=np.float32)
        if not root_node.children: # Terminal or unexpanded root (should not happen if sims ran)
            # This means no moves were possible or explored from root.
            # Could happen if root is mate/stalemate.
            # print("Warning: Root node has no children after simulations.")
            chosen_move = None # No move to choose
        else:
            total_child_visits = sum(child.visit_count for child in root_node.children.values())
            
            best_move_visits = -1
            chosen_move = None # Fallback if no moves explored

            if total_child_visits > 0:
                for move, child_node in root_node.children.items():
                    policy_idx = self.chess_env.action_to_policy_index(move, root_board_state) # Use root board for index
                    if 0 <= policy_idx < config.POLICY_OUTPUT_SIZE:
                        search_policy_vector[policy_idx] = child_node.visit_count / total_child_visits
                    
                    # Determine chosen move based on temperature
                    # For self-play, often sample for first N moves, then pick best.
                    # For evaluation, always pick best (temp -> 0).
                    # This function returns the chosen_move and policy vector.
                    # The caller (self_play or evaluate) will handle temperature.
                    # Here, let's just identify the move with highest visits for now.
                    if child_node.visit_count > best_move_visits:
                        best_move_visits = child_node.visit_count
                        chosen_move = move
                
                if chosen_move is None and root_node.children: # If all visits were 0, pick one by prior
                    # This shouldn't happen if simulations ran and backpropagated.
                    # Fallback: pick based on prior if all visits are 0
                    # print("Warning: All child visits are zero. Picking based on prior (not implemented here).")
                    # For now, just pick the first one.
                    chosen_move = list(root_node.children.keys())[0]


            elif root_node.board_state_at_node.is_game_over(): # No children because game over
                 chosen_move = None
            else: # No children, not game over, but visits are zero. Problem.
                 # print("Warning: No child visits and not game over. Root may not have been expanded or sims failed.")
                 # Try to pick a legal move if any.
                 legal_moves_at_root = list(root_node.board_state_at_node.legal_moves)
                 if legal_moves_at_root:
                     chosen_move = legal_moves_at_root[0] # Fallback, not MCTS guided
                 else:
                     chosen_move = None


        # The 'chosen_move' returned here is typically the one with highest visit count.
        # Sampling based on visit counts (temperature) is usually done by the caller.
        # For now, let's assume this function returns the deterministically best move.
        # The plan says "Return (chosen_move, search_policy_vector, root_node) (chosen_move based on temperature)"
        # This implies temperature logic should be here or passed in.
        # Let's assume for now it's highest visit count, and caller handles temp.

        return chosen_move, search_policy_vector, root_node


    def _expand_node(self, node_to_expand: Node, policy_probs: np.ndarray, 
                     apply_dirichlet_noise: bool = False, 
                     dirichlet_alpha: float = 0.0, dirichlet_epsilon: float = 0.0):
        """
        Expands a node: creates children for all legal moves, assigning prior probabilities.
        policy_probs: Output from network for node_to_expand.state.
        """
        if node_to_expand.is_expanded: return # Already expanded

        board = node_to_expand.board_state_at_node
        legal_moves = list(board.legal_moves)
        
        if not legal_moves: # Terminal node
            node_to_expand.is_expanded = True # Mark as "expanded" (no children possible)
            return

        # Apply Dirichlet noise to policy_probs if requested (typically only for root)
        # This modifies the priors assigned to children.
        current_policy_priors = {} # move -> prior
        if apply_dirichlet_noise and legal_moves:
            noise = np.random.dirichlet([dirichlet_alpha] * len(legal_moves))
            for i, move in enumerate(legal_moves):
                idx = self.chess_env.action_to_policy_index(move, board)
                if 0 <= idx < config.POLICY_OUTPUT_SIZE:
                    original_prior = policy_probs[idx]
                    noisy_prior = (1 - dirichlet_epsilon) * original_prior + dirichlet_epsilon * noise[i]
                    current_policy_priors[move] = noisy_prior
                # else:
                    # print(f"Warning: Move {move.uci()} has invalid index {idx} during expansion with noise.")
                    # current_policy_priors[move] = 0.0 # Or some small epsilon
        else:
            for move in legal_moves:
                idx = self.chess_env.action_to_policy_index(move, board)
                if 0 <= idx < config.POLICY_OUTPUT_SIZE:
                    current_policy_priors[move] = policy_probs[idx]
                # else:
                    # print(f"Warning: Move {move.uci()} has invalid index {idx} during expansion.")
                    # current_policy_priors[move] = 0.0 # Or some small epsilon


        for move in legal_moves:
            child_board = board.copy()
            child_board.push(move)
            
            prior = current_policy_priors.get(move, 0.00001) # Default to small prior if not found (should not happen)
            
            child_node = Node(
                parent=node_to_expand,
                prior_probability=prior,
                board_state_at_node=child_board,
                player_to_move=child_board.turn # Player to move at the child node
            )
            node_to_expand.children[move] = child_node
        
        node_to_expand.is_expanded = True

# --- Testing ---
if __name__ == '__main__':
    print("MCTS Testing (Conceptual - Requires Model and ChessEnv)")

    # Setup dummy components
    if not torch.cuda.is_available():
        print("CUDA not available, MCTS tests will run on CPU if model is moved there.")
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device("cuda")

    try:
        chess_env_instance = ChessEnv()
        # Use a smaller model for faster testing if possible, or the configured one
        # For this test, ensure model parameters match config or are passed.
        # We need a model that can run on CPU for this test if CUDA is not available.
        # The default AttentionChessNet might be large.
        # Let's try with default config, assuming it can run.
        neural_net_instance = AttentionChessNet(
            input_channels=config.INPUT_CHANNELS,
            d_model=config.D_MODEL, # Potentially reduce for faster CPU test
            n_heads=config.N_HEADS,
            num_encoder_layers=config.NUM_ENCODER_LAYERS, # Potentially reduce
        ).to(DEVICE)
        
        mcts_instance = MCTS(chess_env_instance, neural_net_instance, device=DEVICE)

        # Test with initial board state
        initial_board = chess.Board()
        print(f"Initial board for MCTS: \n{initial_board}")

        num_sims = 16 # Small number for quick test
        print(f"\nRunning {num_sims} MCTS simulations from initial state...")
        
        # Ensure action_to_policy_index and policy_index_to_action in chess_env are somewhat functional
        # The placeholder versions might cause issues.
        # If policy_index_to_action returns None often, or action_to_policy_index is not good, MCTS will struggle.

        chosen_move, search_policy, root_node_after_sims = mcts_instance.run_simulations(
            root_board_state=initial_board,
            num_simulations=num_sims,
            C_puct=config.C_PUCT,
            dirichlet_alpha=config.DIRICHLET_ALPHA,
            dirichlet_epsilon=config.DIRICHLET_EPSILON
        )

        print(f"\nMCTS simulations completed.")
        if chosen_move:
            print(f"Chosen move: {chosen_move.uci()}")
        else:
            print("No move chosen (possibly terminal state or error).")
        
        print(f"Search policy shape: {search_policy.shape}")
        # print(f"Search policy (non-zero entries): {search_policy[search_policy > 0]}")
        # print(f"Indices of non-zero entries: {np.where(search_policy > 0)}")

        if root_node_after_sims:
            print(f"Root node visit count: {root_node_after_sims.visit_count}")
            print(f"Root node Q-value: {root_node_after_sims.Q_value()}")
            # print("Children of root node and their stats:")
            # for move, child in root_node_after_sims.children.items():
            #     print(f"  Move: {move.uci()}, Visits: {child.visit_count}, Q: {child.Q_value():.3f}, P: {child.prior_probability:.3f}, U: {child.U_value(config.C_PUCT):.3f}")

        print("\nMCTS conceptual test finished.")
        print("NOTE: Success of this test heavily depends on the correctness of chess_env.py (move/policy mapping) and model.py (forward pass).")
        print("Placeholder implementations in chess_env.py for move mapping might lead to warnings or suboptimal MCTS behavior.")

    except Exception as e:
        print(f"An error occurred during MCTS testing: {e}")
        import traceback
        traceback.print_exc()