import torch.nn as nn
import torch
from omegaconf import DictConfig


def get_model(cfg: DictConfig):
    # Create model based on configuration
    model_kwargs = {k: v for k, v in cfg.model.items() if k != "type"}

    if cfg.model.type == "simple_cnn":
        # Specific kwargs for SimpleCNN
        model_kwargs["n_input_channels"] = len(cfg.data.input_vars)
        model_kwargs["n_output_channels"] = len(cfg.data.output_vars)
        model = SimpleCNN(**model_kwargs)
    elif cfg.model.type == "convlstm":
        # Ensure 'num_output_variables' is set, defaulting from cfg.data if not present in model config
        # as per the ClimateConvLSTM class __init__ and the plan's example.
        if "num_output_variables" not in model_kwargs:
            model_kwargs["num_output_variables"] = len(cfg.data.output_vars)

        # Validate required parameters for ClimateConvLSTM are present in the configuration.
        # These parameters are defined in `configs/model/convlstm.yaml` and loaded into model_kwargs.
        # Explicit checks are good practice, as shown in the plan's `get_model` example.
        required_params = ["input_channels", "sequence_length", "convlstm_hidden_dims", "convlstm_kernel_sizes"]
        for param in required_params:
            if param not in model_kwargs:
                raise ValueError(f"ClimateConvLSTM requires '{param}' in its model configuration.")
        
        # convlstm_bias has a default in the ClimateConvLSTM class constructor.
        model = ClimateConvLSTM(**model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
    return model


# --- Model Architectures ---


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.skip(identity)
        out = self.relu(out)

        return out


class SimpleCNN(nn.Module):
    def __init__(
        self,
        n_input_channels,
        n_output_channels,
        kernel_size=3,
        init_dim=64,
        depth=4,
        dropout_rate=0.2,
    ):
        super().__init__()

        # Initial convolution to expand channels
        self.initial = nn.Sequential(
            nn.Conv2d(n_input_channels, init_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(init_dim),
            nn.ReLU(inplace=True),
        )

        # Residual blocks with increasing feature dimensions
        self.res_blocks = nn.ModuleList()
        current_dim = init_dim

        for i in range(depth):
            out_dim = current_dim * 2 if i < depth - 1 else current_dim
            self.res_blocks.append(ResidualBlock(current_dim, out_dim))
            if i < depth - 1:  # Don't double the final layer
                current_dim *= 2

        # Final prediction layers
        self.dropout = nn.Dropout2d(dropout_rate)
        self.final = nn.Sequential(
            nn.Conv2d(current_dim, current_dim // 2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(current_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(current_dim // 2, n_output_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.initial(x)

        for res_block in self.res_blocks:
            x = res_block(x)

        x = self.dropout(x)
        x = self.final(x)

        return x

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
