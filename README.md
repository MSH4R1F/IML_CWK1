# IML Coursework 1

## Project Setup

If you are totally a beginner, follow these instructions so that you can run my code properly

### Steps:
#### 1. Install Python
- Download and install Python from [here](https://www.python.org/downloads/)
- Make sure to check the box that says "Add Python to PATH" during installation
- Make sure to download the latest version of Python, this would work with Python 3.10

#### 2. Install Git
- Download and install Git from [here](https://git-scm.com/downloads)
- Make sure to check the box that says "Add Git to PATH" during installation

#### 3. Clone the repository
- Open your terminal
- Run the following command to clone the repository
```git clone https://github.com/MSH4R1F/IML_CWK1```

#### 4. Setup the virtual environment
- Navigate to the directory where you cloned the repository
- Run the following command to create a virtual environment
```python -m venv venv```
- Run the following command to activate the virtual environment
```venv\Scripts\activate```
- If you are using a Mac or Linux, run the following command to activate the virtual environment
```source venv/bin/activate```
- Every time you want to run the code, make sure to activate the virtual environment first
#### 5. Install the required packages
- Run the following command to install the required packages
```pip install -r requirements.txt```
- If pip doesn't work, try using pip3
- If pip is not installed at all, you can install it by running the following command
```python get-pip.py```

Follow the next few instructions depending on what you want to do


## Running the code

### Viewing the decision tree plot
To view the decision tree plot for your dataset, follow these steps:

1. Ensure you have the required packages installed as mentioned in the Project Setup section.
2. Open your terminal.
3. Navigate to the directory where your training dataset file is located.
4. Run the following command, replacing `path/to/dataset.txt` with the actual path to your training dataset file:
    ```bash
    python decision_tree.py path/to/dataset.txt
    ```
5. The decision tree will be trained and a plot of the decision tree will be saved as `decision_tree.png` in the current directory.


### Getting the Evaluation metrics
To get the evaluation metrics for your dataset, follow these steps:

1. Ensure you have the required packages installed as mentioned in the Project Setup section.
2. Open your terminal.
3. Navigate to the directory where your training dataset files are located.
4. Run the following command, replacing `path/to/dataset1.txt` and `path/to/dataset2.txt` with the actual paths to your training dataset files:
    ```bash
    python decision_tree_evaluation.py path/to/dataset1.txt path/to/dataset2.txt
    ```
5. If you only have one dataset file, run the following command instead, replacing `path/to/dataset.txt` with the actual path to your training dataset file:
    ```bash
    python decision_tree_evaluation.py path/to/dataset.txt
    ```
6. The evaluation metrics will be printed in the terminal, they may take a few seconds to compute



