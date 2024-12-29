## File Structure

```
EEG_Attention_Classification/
├── data/                     # Directory for raw EEG data and marker files
│   ├── Subject_1/
│   │   ├── Subject_1_oddball_paradigm_eeg.csv
│   │   ├── Subject_1_oddball_paradigm_markers.csv
│   │   ├── ...
│   ├── Subject_2/
│   │   ├── ...
│
├── src/                      # Source code directory
│   ├── data_loader.py        # Functions to load EEG and marker data
│   ├── preprocessing.py      # Preprocessing utilities (e.g., filtering, epoching)
│   ├── feature_extraction.py # Feature extraction methods
│   ├── model.py              # ML models and training pipeline
│   ├── evaluation.py         # Evaluation metrics and visualization
│
├── notebooks/                # Jupyter notebooks for experiments and analysis
│   ├── EDA.ipynb             # Exploratory Data Analysis notebook
│
├── output/                   # Outputs (e.g., models, logs, and reports)
│   ├── models/               # Saved trained models
│   ├── results/              # Evaluation results
│
├── main.py                   # Main script to run the pipeline
├── requirements.txt          # Python dependencies
├── README.md                 # Project description and instructions
```


```
docker build -t eeg-project .
```

```
docker run --rm -it -v "${PWD}/data:/app/data" eeg-project
```