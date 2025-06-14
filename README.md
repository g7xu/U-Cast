# U-Cast

## 1. Intro
This project focuses on climate emulation, a machine learning approach to predict future climate patterns. It addresses the computational expense of traditional physics-based climate models by developing alternative models that can maintain prediction accuracy. The task is significant for generating actionable climate predictions and understanding Earth's future climate under various emissions scenarios.

## Competition Details
This is also a competition that was being held on Kaggle named CSE 151B Competition Spring 2025 - Climate Emulation on Kaggle.
*   **Specific Machine Learning Task:** Develop models to predict future climate variables (surface air temperature `tas` and precipitation rate `pr`) based on historical data and input forcings (`CO2`, `CH4`, `SO2`, `BC`, `rsdt`) under new Shared Socioeconomic Pathway (SSP) scenarios.
*   **Goal:** The primary objective is to develop accurate climate emulation models and achieve the best possible prediction accuracy on the competition metrics (Monthly Area-Weighted RMSE, Decadal Mean Area-Weighted RMSE, and Decadal Standard Deviation Area-Weighted MAE).

## 3. Project Setup and Installation

*   **Prerequisites:** download all the dependencies in the requirement.txt file
*   **Installation:** Provide clear instructions on how to set up the project environment. This typically involves cloning the repository and installing dependencies using `requirements.txt` or `pyproject.toml`.
    ```bash
    git clone <repository_url>
    cd U-cast
    pip install -r requirements.txt
    # or using poetry
    # poetry install
    ```
## 4. Data

*   **Data Source:** The data is provided by the CSE 151B Competition on Kaggle, originating from CMIP6 climate model simulations under different Shared Socioeconomic Pathway (SSP) scenarios.
*   **Data Description:** The dataset contains monthly climate variables (precipitation and temperature) and input variables (forcings) like CO2, SO2, CH4, BC, and rsdt. The data is coarsened to a (48, 72) lat-lon grid and includes a time dimension (monthly data) and a member ID dimension (3-member ensemble for each scenario). Training data is from SSP126, SSP370, and SSP585 scenarios. Validation data is the last 10 years of SSP370, and testing data is from SSP245. The dataset is stored in Zarr format.
*   **Data Processing:** The provided starter code in `main.py` includes preprocessing steps such as Z-score normalization of input and output variables and broadcasting of global input variables to match spatial dimensions. These steps can be modified to suit different model architectures and requirements.

## 5. Experiments and Model Development

*   **Overview of Models Tried:** We experimented with several model architectures, including a Simple CNN baseline, a ConvLSTM network to capture temporal dependencies, Vision Transformers (ViTs), and an Enhanced Climate UNet.
*   **Experiment Results:**
    *   **Evaluation Metrics:** We primarily used the competition's area-weighted metrics (Monthly Area-Weighted RMSE, Decadal Mean Area-Weighted RMSE, and Decadal Standard Deviation Area-Weighted MAE) for final evaluation on Kaggle. During development, we also monitored training speed per epoch, convergence rate based on loss curves, and signs of overfitting by comparing validation loss against training loss.
    *   **Training Speed:** Simple CNN (~1 min/epoch), ConvLSTM (~2.5 min/epoch), U-Net (~3 min/epoch).
    *   **Convergence and Overfitting:** We observed that the Adam optimizer helped achieve faster convergence compared to the initial optimizer. We monitored validation loss to prevent overfitting, especially given the limited dataset size.
    *   **Hyperparameter Tuning:** Key hyperparameters tuned included learning rate (exploring 0.0001, 0.00005, 0.00003), kernel size (3, 5, 7), network depth (2, 4, 6), and dropout rate (0.01 to 0.1).
    *   **Kaggle Leaderboard:** Our final U-Net model significantly outperformed the baselines and ViT on both the public and private Kaggle leaderboards (see Section 8 for detailed results).

## 6. Final Model

*   **Final Model Selection:** The Enhanced Climate UNet was selected as the final model due to its strong performance on the validation set and Kaggle leaderboard, faster training convergence, and better generalization compared to the Simple CNN, ConvLSTM, and Vision Transformer models we experimented with. Its U-Net architecture with skip connections proved effective in capturing multi-scale spatial features crucial for climate emulation.
*   **Model Architecture:** The final model is a U-Net-style Convolutional Neural Network.
    *   **CoordConv2d:** The input layer incorporates CoordConv2d to explicitly provide spatial coordinates (latitude and longitude) to the network, enabling it to learn geographically aware patterns.
    *   **Encoder:** The encoder path consists of several levels, each using ResidualSEBlocks followed by a 2x2 stride-2 convolution for downsampling. This progressively extracts hierarchical spatial features at reduced resolutions.
    *   **ResidualSEBlock:** A custom block combining residual connections, Group Normalization, GELU activation, and a Squeeze-and-Excitation (SE) module for adaptive channel reweighting.
    *   **Bottleneck:** The bottleneck uses stacked ResidualSEBlocks with dilated convolutions (rates 1, 2, and 4) to expand the receptive field without losing spatial resolution, followed by a Dropout2d layer (rate 0.2) for regularization.
    *   **Decoder:** The decoder path mirrors the encoder, using transposed convolutions for upsampling and concatenating skip connections from corresponding encoder levels. ResidualSEBlocks are used to refine the combined features and recover fine-scale spatial detail. Bilinear interpolation is used in the final upsampling step to match the original width (72).
    *   **Final Projection:** A 1x1 convolution maps the final feature map to the 2 output channels (surface air temperature and precipitation rate).
*   **Training Details:**
    *   **Optimizer:** Adam optimizer was used for training.
    *   **Batch Size:** A batch size of 42 was used due to computational constraints.
    *   **Training Duration:** Training duration varied by model complexity (Simple CNN: ~1 min/epoch, ConvLSTM: ~2.5 min/epoch, U-Net: ~3 min/epoch).
*   **Final Parameters:** The final configuration for the best performing U-Net model included a kernel size of 5, a network depth of 4, and a dropout rate of 0.0422.

## 8. Results and Evaluation

*   Present the final performance of your chosen model on the evaluation dataset using the competition metric.
*   Optionally, include visualizations or charts to illustrate the results.

## 9. Repository Structure

```
U-Cast/
├── .gitignore
├── README.md
├── requirements.txt
├── pyproject.toml
├── Makefile
├── _climate_kaggle_metric.py
├── _test_kaggle_metric.py
├── main.py
├── configs/
│   ├── main_config.yaml
│   ├── data/
│   │   └── default.yaml
│   ├── model/
│   │   ├── convlstm.yaml
│   │   ├── enhanced_climate_unet.yaml
│   │   └── simple_cnn.yaml
│   ├── trainer/
│   │   └── default.yaml
│   └── training/
│       └── default.yaml
├── notebooks/
│   ├── data-exploration-basic.ipynb
│   └── extended_data_exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── models.py
│   └── utils.py
└── submissions/
```

## 10. Acknowledgements

*   Guoxuan Xu
*   Angela Hu
*   Ciro Zhang