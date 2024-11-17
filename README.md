# Predict Language from Phonemes

This project involves a neural network model that predicts the language of a word based on its phoneme sequence. The model is trained using words in multiple languages, with phonemes as the input features.

---

## Features
- **Language Support:** Predicts among multiple languages: English (`uk`), German (`gr`), Czech (`cs`), and French (`fr`).
- **Configurable Word Length:** Allows setting the word length for training and testing (default: 5).
- **Neural Network:** Fully connected network with embedding layers and two hidden layers.
- **Training and Testing Pipelines:** Supports both training on a subset of data and evaluating accuracy on a test set.
- **Phoneme Analytics:** Extracts phoneme lists and calculates frequency counts for each language.
- **Preprocessing Utility:** Creates word-length-specific files from raw input word lists.

---

## **Python Scripts**

### 1. `create_n_letter_word_list.py`
- **Description**: A script to generate word lists of a specific length based on phonetic rules for different languages.
- **Usage**: Run this script to create custom word lists from the provided data files.

### 2. `main.py`
- **Description**: The main entry point for the repository, combining functionalities from other scripts. Likely designed to orchestrate processes such as loading datasets, analyzing phonetics, or generating word lists.

---

## **Language-Specific Word Data**

### **Czech (`cs_`)**
1. `cs_phon_len4.txt` - Words with 4 phonemes in Czech.
2. `cs_phon_len5.txt` - Words with 5 phonemes in Czech.
3. `cs_phon_len6.txt` - Words with 6 phonemes in Czech.
4. `cs_phon_len7.txt` - Words with 7 phonemes in Czech.
5. `cs_phon_len8.txt` - Words with 8 phonemes in Czech.
6. `cs_phon_len9.txt` - Words with 9 phonemes in Czech.
7. `cs_phon_len10.txt` - Words with 10 phonemes in Czech.
8. `cs_phon_len11.txt` - Words with 11 phonemes in Czech.
9. `cs_words.txt` - The complete dataset of Czech words.

### **French (`fr_`)**
1. `fr_phon_len4.txt` - Words with 4 phonemes in French.
2. `fr_phon_len5.txt` - Words with 5 phonemes in French.
3. `fr_phon_len6.txt` - Words with 6 phonemes in French.
4. `fr_phon_len7.txt` - Words with 7 phonemes in French.
5. `fr_phon_len8.txt` - Words with 8 phonemes in French.
6. `fr_phon_len9.txt` - Words with 9 phonemes in French.
7. `fr_phon_len10.txt` - Words with 10 phonemes in French.
8. `fr_phon_len11.txt` - Words with 11 phonemes in French.
9. `fr_words.txt` - The complete dataset of French words.

### **German (`gr_`)**
1. `gr_phon_len4.txt` - Words with 4 phonemes in German.
2. `gr_phon_len5.txt` - Words with 5 phonemes in German.
3. `gr_phon_len6.txt` - Words with 6 phonemes in German.
4. `gr_phon_len7.txt` - Words with 7 phonemes in German.
5. `gr_phon_len8.txt` - Words with 8 phonemes in German.
6. `gr_phon_len9.txt` - Words with 9 phonemes in German.
7. `gr_phon_len10.txt` - Words with 10 phonemes in German.
8. `gr_phon_len11.txt` - Words with 11 phonemes in German.
9. `gr_words.txt` - The complete dataset of German words.

### **Ukrainian (`uk_`)**
1. `uk_phon_len4.txt` - Words with 4 phonemes in Ukrainian.
2. `uk_phon_len5.txt` - Words with 5 phonemes in Ukrainian.
3. `uk_phon_len6.txt` - Words with 6 phonemes in Ukrainian.
4. `uk_phon_len7.txt` - Words with 7 phonemes in Ukrainian.
5. `uk_phon_len8.txt` - Words with 8 phonemes in Ukrainian.
6. `uk_phon_len9.txt` - Words with 9 phonemes in Ukrainian.
7. `uk_phon_len10.txt` - Words with 10 phonemes in Ukrainian.
8. `uk_phon_len11.txt` - Words with 11 phonemes in Ukrainian.
9. `uk_words.txt` - The complete dataset of Ukrainian words.

---

## Requirements

To run this project, you'll need the following libraries:
- Python 3.x
- PyTorch
- NumPy

Install the required packages by running the following:

   ```bash pip install torch numpy

## Setup and Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/phoneme-prediction-nn.git
   cd phoneme-prediction-nn

## Running the Scripts
1. Use `create_n_letter_word_list.py` to generate word lists based on specific conditions.
   ```bash
   python create_n_letter_word_list.py

2. Use `python main.py` to create NN and predict phonemes