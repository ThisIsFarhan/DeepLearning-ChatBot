# Chatbot with Intent Prediction and FastAPI Integration

This repository contains a chatbot built using JSON data, where a neural network is employed to predict user intent and generate responses accordingly. The model processes user input, predicts the intent, and then provides an appropriate response. Additionally, a FastAPI backend is set up to handle requests and output the response via an API.

## Features

- **Neural Network for Intent Prediction**: The chatbot uses a neural network to classify the user's intent based on input text.
- **Response Generation**: Once the intent is predicted, the corresponding response is retrieved from a pre-defined set of responses stored in JSON format.
- **FastAPI Backend**: A FastAPI API is provided to serve the chatbot, where users can send POST requests and receive responses in real-time.
- **JSON Data for Easy Customization**: All intents and responses are stored in a JSON file for easy customization and extension.

## Installation

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/ThisIsFarhan/DeepLearning-ChatBot
    ```

2. Navigate to the project directory:

    ```bash
    cd DeepLearning-ChatBot
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Start the FastAPI server

To run the chatbot API server:

```bash
uvicorn app:app --reload
