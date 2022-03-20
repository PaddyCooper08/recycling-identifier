# recycling-identifier
There are 2 parts to this project **ML** (the machine learning bit) and **cli** (the interactive bit).
Currently the dataset for the machine learning section is not very accurate (~30%) because I don't have enough data. Therefore, I have decided to leave this bit out of the current build. This code will run on any machine including a raspberry pi, mac and windows computer.
There is a website in the **very** early stages (it's just a header) found at github.com/PaddyCooper08/rf-website

# Getting started
These instructions are for a windows computer, you may need to change the commands to match your os.
### For the dataset:
1. Clone the git repo to your local machine (this requires git to be installed on your computer):
    >`git clone "https://github.com/PaddyCooper08/recycling-identifier"`
2. Create a new virtual environment to house the dependencies (this could take a while):
   > `python3 -m venv ./venv`
3. Activate said virtual environment:
    >`cd venv/scripts/ && activate`
4. Install the dependencies (this will take quite a long time):
    >`pip install -r requirements.txt` (you need to be in the same directory as requirements.txt)
5. Build the data model:
    > `cd ./ML` <br />
    `python3 global.py` <br />
    Wait for this to finish then: <br />
    `python3 train_test.py` <br />
    You will be shown a graph with the accuracies and the model is stored in ML/output.
    
### For the CLI:
1. cd into the correct folder:
    > `cd cli/paddy-recycling-identifier/paddy_recycling_identifier` 
## Commands:
| Command         | Arguments     | 
|--------------|-----------|
| --help      | n/a
| recycle      | Numbers 1-7  |
See the showcase video for a better tutorial.

# To do
1. Create website/desktop app to replace cli
2. Improve dataset accuracy
3. When more accurate, implement dataset
