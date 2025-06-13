# U-Cast

## 1. Intro

*   **Context of the Project:** Briefly introduce the problem domain (climate data prediction) and the significance of the task.
*   **Competition Details:**
    *   What is the competition? (Mention the name and platform, e.g., Kaggle CSE 151B Spring 2025 Competition).
    *   What is the specific machine learning task? (e.g., predicting future climate variables based on historical data).
*   **Goal:** State the primary objective of the project (e.g., achieve the best possible prediction accuracy on the competition metric).



## 3. Project Setup and Installation

*   **Prerequisites:** List any necessary software or dependencies (e.g., Python, specific libraries).
*   **Installation:** Provide clear instructions on how to set up the project environment. This typically involves cloning the repository and installing dependencies using `requirements.txt` or `pyproject.toml`.
    ```bash
    git clone <repository_url>
    cd climate-data-prediction-cse151b
    pip install -r requirements.txt
    # or using poetry
    # poetry install
    ```

## 4. Data

*   **Data Source:** Briefly describe where the data comes from (e.g., provided by the competition).
*   **Data Description:** Explain the dataset, including the types of data, features, and target variables. Mention the size and format of the data.
*   **Data Processing:** Detail the steps taken to preprocess the data, such as cleaning, normalization, feature engineering, and splitting the data.

## 5. Experiments and Model Development

*   **Overview of Models Tried:** List and briefly describe the different models or architectures you experimented with (e.g., Simple CNN, ConvLSTM, Enhanced Climate UNet).
*   **Experiment Results:** For each model, present the key results, including:
    *   Accuracy or relevant evaluation metrics (mention the metric used, e.g., RMSE, MAE).
    *   Key parameters used for training.
    *   Mention any significant findings or challenges encountered during experimentation.

## 6. Final Model

*   **Final Model Selection:** Explain why the chosen model was selected as the final solution.
*   **Model Architecture:** Provide a more detailed description of the final model's architecture.
*   **Training Details:** Describe the training process for the final model, including hyperparameters, optimizer, loss function, and number of epochs.
*   **Final Parameters:** Specify the final parameters or configuration used for the best performing model.

## 7. How to Run

*   Provide instructions on how to run the main scripts for training, evaluation, or prediction.
    ```bash
    # Example command to run training
    python main.py --config configs/training/default.yaml
    ```
*   Explain any necessary command-line arguments or configuration files.

## 8. Results and Evaluation

*   Present the final performance of your chosen model on the evaluation dataset using the competition metric.
*   Optionally, include visualizations or charts to illustrate the results.

## 9. Repository Structure

    ```
    climate-data-prediction-cse151b/
    ├── .gitignore
    ├── README.md
    ├── requirements.txt
    ├── pyproject.toml
    ├── Makefile
    ├── _climate_kaggle_metric.py
    ├── _test_kaggle_metric.py
    ├── main.py
    ├── test.py
    │
    ├── configs/
    │   └── all the configuration files
    │
    ├── notebooks/
    │   ├── data-exploration-basic.ipynb (basic exploration)
    │   └── extended_data_exploration.ipynb (detailed exploration)
    │
    └── src/
        ├── __init__.py
        ├── models.py (implementation of all the models)
        └── utils.py
    ```

## 10. Future Work (Optional)

*   Suggest potential improvements or future directions for the project.


## 1. Acknowledgements

*   List the team members who contributed to the project.