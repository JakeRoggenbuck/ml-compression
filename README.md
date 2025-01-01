# ml-compression

Automatically create compression schemes convolutional neural networks

## Description

Train a neural network to have its output reproduce the exact input. Why is this interesting? You can constrict the quantity of neurons in one of the intermediate layer and use this as a stand-in for the data that was input. This then means that you have effectively compressed this data.

These types of models can help us generate optimal compression schemes.

## Setup
```
pip install -r requirements.txt
```

## Running
```
python3 main.py
```

## Results

![image](https://github.com/user-attachments/assets/86069543-ec8c-46b8-a559-116e5f2c77b4)
