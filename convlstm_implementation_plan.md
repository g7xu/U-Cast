# Plan for ConvLSTM Model Integration

**Overall Goal:** Implement the `ClimateConvLSTM` model as specified in the design document, making it configurable and usable within your existing PyTorch Lightning framework.

**Key Files to Modify/Create:**

1.  **`src/models.py`**: Add `CustomConvLSTMCell` and `ClimateConvLSTM` classes, and update `get_model`.
2.  **`configs/model/convlstm.yaml`** (New File): Define configuration for the `ClimateConvLSTM` model.
3.  **(Conceptual) `src/data_module.py` or equivalent**: Ensure data loading provides sequences in the correct shape.
4.  **(Conceptual) `main.py` or training script**: Update to allow selection of the `convlstm` model.

---

### Step 1: Create Model Configuration File

Create a new YAML file: `configs/model/convlstm.yaml`

```yaml
# @package _group_
# Configuration for the ClimateConvLSTM model

type: "convlstm" # Model type identifier for get_model

# Parameters for ClimateConvLSTM.__init__
input_channels: 8      # As per design: 5 forcings + 2 seasonal + 1 latitude
num_output_variables: 2 # For 'tas' and 'pr'
sequence_length: 12    # Example sequence length, can be tuned

# Parameters for CustomConvLSTMCell layers
convlstm_hidden_dims: [64, 128] # Hidden dimensions for the two ConvLSTM layers
convlstm_kernel_sizes: [[3,3], [3,3]] # Kernel sizes for the two ConvLSTM layers
convlstm_bias: True

```

**Rationale:**
*   This file centralizes all hyperparameters for the `ClimateConvLSTM` model.
*   `type: "convlstm"` will be used by the `get_model` factory function.
*   `input_channels: 8` explicitly sets the expected number of input channels.
*   `sequence_length` is a crucial hyperparameter for sequence models.
*   `convlstm_hidden_dims` and `convlstm_kernel_sizes` allow configuration of the ConvLSTM layers.

---

### Step 2: Update `src/models.py`

#### 2.1. Add `CustomConvLSTMCell` Class

Add the `CustomConvLSTMCell` class definition as provided in the design document to `src/models.py`.

```python
# (Existing imports ...)
# import torch # Ensure torch is imported
# import torch.nn as nn # Ensure nn is imported

# ... (Existing ResidualBlock and SimpleCNN classes) ...

# Add the CustomConvLSTMCell class here:
class CustomConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(CustomConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        # Calculate padding for 'same' output
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim, # For i, f, o, g gates
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        # Split into gates: input, forget, output, cell gate
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))

```

#### 2.2. Add `ClimateConvLSTM` Class

Add the `ClimateConvLSTM` class definition as provided in the design document to `src/models.py`, adapting it to take parameters from the configuration.

```python
# ... (After CustomConvLSTMCell) ...

class ClimateConvLSTM(nn.Module):
    def __init__(self, input_channels, num_output_variables, sequence_length,
                 convlstm_hidden_dims, convlstm_kernel_sizes, convlstm_bias=True):
        super(ClimateConvLSTM, self).__init__()
        self.sequence_length = sequence_length
        self.input_channels = input_channels

        # --- ConvLSTM Block 1 ---
        self.convlstm_cell1 = CustomConvLSTMCell(
            input_dim=self.input_channels,
            hidden_dim=convlstm_hidden_dims[0],
            kernel_size=tuple(convlstm_kernel_sizes[0]),
            bias=convlstm_bias
        )
        self.bn1 = nn.BatchNorm2d(num_features=convlstm_hidden_dims[0])

        # --- ConvLSTM Block 2 ---
        self.convlstm_cell2 = CustomConvLSTMCell(
            input_dim=convlstm_hidden_dims[0],
            hidden_dim=convlstm_hidden_dims[1],
            kernel_size=tuple(convlstm_kernel_sizes[1]),
            bias=convlstm_bias
        )
        self.bn2 = nn.BatchNorm2d(num_features=convlstm_hidden_dims[1])

        # --- Output Prediction Head ---
        self.output_conv = nn.Conv2d(
            in_channels=convlstm_hidden_dims[1],
            out_channels=num_output_variables,
            kernel_size=(1, 1),
            padding=0
        )

    def forward(self, x_sequence):
        # x_sequence shape: (batch_size, sequence_length, C_in, height, width)
        batch_size, seq_len, _, H, W = x_sequence.shape

        # Initialize hidden states for each layer
        h1, c1 = self.convlstm_cell1.init_hidden(batch_size, (H, W), x_sequence.device)
        h2, c2 = self.convlstm_cell2.init_hidden(batch_size, (H, W), x_sequence.device)

        # Loop over sequence length
        for t in range(seq_len):
            x_t = x_sequence[:, t, :, :, :]
            h1, c1 = self.convlstm_cell1(x_t, (h1, c1))
            h1_bn = self.bn1(h1)

            h2, c2 = self.convlstm_cell2(h1_bn, (h2, c2))
            h2_bn = self.bn2(h2)

        last_hidden_state_final = h2_bn
        prediction = self.output_conv(last_hidden_state_final)

        return prediction

```

#### 2.3. Update `get_model` Function

Modify the `get_model` function in `src/models.py` to handle the new `"convlstm"` type and its specific parameters.

```python
# def get_model(cfg: DictConfig):
#     # Create model based on configuration
#     model_kwargs = {k: v for k, v in cfg.model.items() if k != "type"}
#
#     if cfg.model.type == "simple_cnn":
#         model_kwargs["n_input_channels"] = len(cfg.data.input_vars)
#         model_kwargs["n_output_channels"] = len(cfg.data.output_vars)
#         model = SimpleCNN(**model_kwargs)
#     elif cfg.model.type == "convlstm":
#         if "input_channels" not in model_kwargs:
#             raise ValueError("ClimateConvLSTM requires 'input_channels' in its model configuration.")
#         if "num_output_variables" not in model_kwargs:
#             model_kwargs["num_output_variables"] = len(cfg.data.output_vars)
#         if "sequence_length" not in model_kwargs:
#             raise ValueError("ClimateConvLSTM requires 'sequence_length' in its model configuration.")
#         if "convlstm_hidden_dims" not in model_kwargs:
#             raise ValueError("ClimateConvLSTM requires 'convlstm_hidden_dims' in its model configuration.")
#         if "convlstm_kernel_sizes" not in model_kwargs:
#             raise ValueError("ClimateConvLSTM requires 'convlstm_kernel_sizes' in its model configuration.")
#
#         model = ClimateConvLSTM(**model_kwargs)
#     else:
#         raise ValueError(f"Unknown model type: {cfg.model.type}")
#     return model
```

**Rationale:**
*   Integrates the custom ConvLSTM cell and the main ConvLSTM model architecture.
*   The `ClimateConvLSTM` constructor is adapted to take hyperparameters from the configuration file.
*   `get_model` is extended to be a factory for `ClimateConvLSTM`.

---

### Step 3: Data Module Considerations (Conceptual)

*   **Input Data Shape:** The `DataModule` must yield `x_sequence` with shape `(batch_size, S, C_in, height, width)`.
*   **Feature Engineering:** The `DataModule` handles:
    *   Loading 5 base forcing variables.
    *   Broadcasting them to `(1, 48, 72)`.
    *   Generating seasonal encodings (`month_sin`, `month_cos`) and broadcasting.
    *   Generating normalized latitude grid.
    *   Concatenating into `(C_in, 48, 72)` per time step.
    *   Assembling sequences of `S` time steps.
*   **Output Data Shape:** Target `y` with shape `(batch_size, num_output_variables, height, width)`.

---

### Step 4: Training Script Considerations (Conceptual)

*   Main training script (e.g., `main.py`) should allow `convlstm` model selection via Hydra:
    `python main.py model=convlstm`

---

### Step 5: Visualization (Mermaid Diagram)

```mermaid
graph TD
    subgraph Input Data Processing (DataModule)
        A[Raw Forcings (5 channels, scalar)] --> B{Broadcast to (1,48,72)}
        C[Month of Year] --> D{Seasonal Encoding (sin, cos)}
        D --> E{Broadcast to (1,48,72)}
        F[Latitude Grid (48,72)] --> G{Normalize Latitude}
        B --> H[Concatenated Input X_t (8, 48, 72)]
        E --> H
        G --> H
        H --> I{Assemble Sequence (S, 8, 48, 72)}
    end

    subgraph ClimateConvLSTM Model
        X_seq[Input: x_sequence (B, S, 8, H, W)] --> Layer1_Loop{Loop t=0 to S-1}

        subgraph Layer1_Loop [ConvLSTM Layer 1 Loop]
            X_t_loop[x_sequence[:, t, :, :, :]] --> Cell1[CustomConvLSTMCell1 (h_dim1=64)]
            Prev_h1c1[h1, c1 (prev_t)] --> Cell1
            Cell1 --> Next_h1c1[h1, c1 (curr_t)]
            Next_h1c1 -- h1 --> BN1[BatchNorm2d(64)]
            BN1 --> h1_bn_out[h1_bn (output for next layer / current t)]
        end

        h1_bn_out --> Layer2_Loop{Loop t=0 to S-1 (conceptually, uses h1_bn from Layer1's loop)}

        subgraph Layer2_Loop [ConvLSTM Layer 2 Loop]
            h1_bn_t[h1_bn from Layer1 at time t] --> Cell2[CustomConvLSTMCell2 (h_dim2=128)]
            Prev_h2c2[h2, c2 (prev_t)] --> Cell2
            Cell2 --> Next_h2c2[h2, c2 (curr_t)]
            Next_h2c2 -- h2 --> BN2[BatchNorm2d(128)]
            BN2 --> h2_bn_out[h2_bn (output / current t)]
        end

        Layer2_Loop -- Last h2_bn (at t=S-1) --> OutputConv[Conv2d (out_channels=2, ks=1x1)]
        OutputConv --> Prediction[Output: (B, 2, H, W)]
    end

    I --> X_seq

    style A fill:#lightgrey,stroke:#333,stroke-width:2px
    style C fill:#lightgrey,stroke:#333,stroke-width:2px
    style F fill:#lightgrey,stroke:#333,stroke-width:2px
    style X_seq fill:#lightblue,stroke:#333,stroke-width:2px
    style Prediction fill:#lightgreen,stroke:#333,stroke-width:2px