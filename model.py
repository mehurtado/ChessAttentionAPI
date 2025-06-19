# model.py
# Phase 2: Core Modules Implementation
# Module 2: Neural Network Architecture (Attention-Based)
# Objective: Define the policy-value network.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
import math

import config # Assuming your config.py

class AttentionChessNet(nn.Module):
    def __init__(self,
                 input_channels: int = config.INPUT_CHANNELS,
                 d_model: int = config.D_MODEL,
                 n_heads: int = config.N_HEADS,
                 num_encoder_layers: int = config.NUM_ENCODER_LAYERS,
                 dim_feedforward_scale: int = config.DIM_FEEDFORWARD_SCALE,
                 dropout_rate: float = config.DROPOUT_RATE,
                 num_policy_outputs: int = config.POLICY_OUTPUT_SIZE, # 8*8*73 = 4672
                 board_size: int = config.BOARD_SIZE):
        super().__init__()

        self.d_model = d_model
        self.board_size = board_size
        self.num_squares = board_size * board_size

        # Stem: Conv2D -> BatchNorm -> ReLU
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU()
        )

        # Positional Encoding (Learned)
        # Input to transformer is (Batch, Seq_Len, EmbeddingDim)
        # Here, Seq_Len is num_squares (64)
        self.learned_pos_encoding = nn.Parameter(torch.randn(1, self.num_squares, d_model))

        # Transformer Encoder Body
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * dim_feedforward_scale,
            dropout=dropout_rate,
            activation='relu',
            batch_first=True, # Expects (Batch, Seq, Feature)
            norm_first=True   # Pre-LN variant, common in modern transformers
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Policy Head
        # Input: (B, num_squares, d_model). Reshape to (B, d_model, board_size, board_size)
        self.policy_head_conv1 = nn.Conv2d(d_model, 128, kernel_size=1, bias=False)
        self.policy_head_bn1 = nn.BatchNorm2d(128)
        self.policy_head_relu = nn.ReLU()
        self.policy_head_conv2 = nn.Conv2d(128, 73, kernel_size=1) # 73 action planes
        # Output will be (B, 73, board_size, board_size). Flatten to (B, 73 * board_size * board_size)

        # Value Head
        # Input: (B, num_squares, d_model)
        # Option 1: Use embedding of a [CLS] token (if one was added, not typical for this setup)
        # Option 2: Global Average Pooling over sequence dimension
        # Option 3: Take the embedding of the first token (pos 0) - simpler
        self.value_head_fc1 = nn.Linear(d_model, 256)
        self.value_head_relu = nn.ReLU()
        self.value_head_fc2 = nn.Linear(256, 1)
        self.value_head_tanh = nn.Tanh()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Parameter) and m is self.learned_pos_encoding: # Check if it's the pos encoding
                 nn.init.normal_(m, std=0.02) # Common for embeddings


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, input_channels, board_size, board_size)
        
        # Stem
        x = self.stem(x)  # (B, d_model, board_size, board_size)
        
        # Reshape for Transformer: (B, d_model, num_squares) -> (B, num_squares, d_model)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1) # (B, num_squares, d_model)
        
        # Add Positional Encoding
        x = x + self.learned_pos_encoding # Broadcasting (1, num_squares, d_model)
        
        # Transformer Encoder Body
        transformer_output = self.transformer_encoder(x) # (B, num_squares, d_model)
        
        # Policy Head
        # Reshape back to (B, d_model, board_size, board_size) for Conv2D
        policy_input = transformer_output.permute(0, 2, 1).view(B, self.d_model, H, W)
        ph = self.policy_head_relu(self.policy_head_bn1(self.policy_head_conv1(policy_input)))
        policy_logits = self.policy_head_conv2(ph) # (B, 73, board_size, board_size)
        policy_logits = policy_logits.view(B, -1) # Flatten to (B, 73 * board_size * board_size) = (B, 4672)
        
        # Value Head
        # Using global average pooling of transformer_output
        # value_input = transformer_output.mean(dim=1) # (B, d_model)
        # Or using the first token's output (if it's treated like a CLS token)
        value_input = transformer_output[:, 0, :] # (B, d_model) - assuming pos 0 is representative
        
        vh = self.value_head_relu(self.value_head_fc1(value_input))
        value = self.value_head_tanh(self.value_head_fc2(vh)) # (B, 1)
        
        return policy_logits, value

# --- Testing ---
if __name__ == '__main__':
    # Test with dummy data
    batch_size = 4
    dummy_input = torch.randn(batch_size, config.INPUT_CHANNELS, config.BOARD_SIZE, config.BOARD_SIZE)
    
    model = AttentionChessNet(
        input_channels=config.INPUT_CHANNELS,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        dim_feedforward_scale=config.DIM_FEEDFORWARD_SCALE,
        dropout_rate=config.DROPOUT_RATE,
        num_policy_outputs=config.POLICY_OUTPUT_SIZE,
        board_size=config.BOARD_SIZE
    )
    
    print(f"Model initialized. Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Test forward pass
    model.eval() # Set to evaluation mode for dropout, batchnorm
    with torch.no_grad():
        policy_logits, value = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Policy logits output shape: {policy_logits.shape}") # Expected: (B, 4672)
    print(f"Value output shape: {value.shape}")          # Expected: (B, 1)

    assert policy_logits.shape == (batch_size, config.POLICY_OUTPUT_SIZE), "Policy output shape mismatch"
    assert value.shape == (batch_size, 1), "Value output shape mismatch"

    print("\nModel forward pass test successful.")
    print("\n--- Minimal JIT Scripting Test ---")
    try:
        scripted_model_test = torch.jit.script(model)
        print("JIT SCRIPTING SUCCEEDED")
    except Exception as e:
        print(f"JIT SCRIPTING FAILED: {e}")

    # Check VRAM usage (conceptual - requires GPU and tools like nvidia-smi or PyTorch utils)
    if torch.cuda.is_available():
        print("\nAttempting to move model and data to GPU for a conceptual VRAM check...")
        device = torch.device("cuda")
        model.to(device)
        dummy_input_gpu = dummy_input.to(device)
        
        # Perform a forward pass on GPU
        policy_logits_gpu, value_gpu = model(dummy_input_gpu)
        print("Forward pass on GPU successful.")
        
        # To check actual VRAM, you'd typically use:
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        # Or monitor nvidia-smi while the script runs.
        # This test just confirms it runs on GPU.
        
        # Example of mixed precision (conceptual, full setup in train.py)
        # from torch.cuda.amp import autocast
        # with autocast():
        #    policy_logits_amp, value_amp = model(dummy_input_gpu)
        # print("Forward pass with autocast (mixed precision) on GPU successful.")

    else:
        print("\nCUDA not available. Skipping GPU tests.")