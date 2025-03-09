# Language Model - Poirot Investigates Text Generation

This repository contains the implementation of a word-level language model trained on the text of *Poirot Investigates* by Agatha Christie. The goal of the project was to create a model capable of predicting the next word in a given sequence and generating text that mimics the writing style of the author.

## Project Overview

The language model was developed using a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers. The model was trained on the text from *Poirot Investigates*, preprocessed and tokenized to predict the next word in a sequence and generate stylistically consistent text.

### Key Steps:
1. **Data Preprocessing**: Text was cleaned and tokenized, with metadata removed and special characters stripped. GloVe word embeddings were used to improve word representation.
2. **Model Architecture**: Built using LSTM layers with embedding and dense layers. Early stopping was applied to prevent overfitting.
3. **Training & Evaluation**: The model was trained for 250 epochs, with performance evaluated based on accuracy and the quality of generated text.

## Dataset

The dataset was extracted from the Project Gutenberg text of *Poirot Investigates*.

- **Text Cleaning**: Removed special characters and punctuation, converted to lowercase.
- **Tokenization**: Tokenized the text into words, limiting the vocabulary size to 30,000 words.
- **GloVe Embeddings**: Pretrained GloVe 100-dimensional word embeddings were used for better word representation.

## Model Architecture

The model was built with the following architecture:

1. **Embedding Layer**: Converts words into dense vectors, initialized with GloVe embeddings.
2. **LSTM Layers**: 
   - First LSTM layer with 256 units to capture long-term dependencies.
   - Second LSTM layer with 128 units to refine learned patterns.
3. **LayerNormalization & Dropout**: To stabilize training and reduce overfitting.
4. **Dense Layer**: A fully connected layer with 128 units, followed by a softmax output to predict the next word.

### Hyperparameters:
- Optimizer: Adam (learning rate = 0.0003)
- Batch Size: 64
- Sequence Length: 100
- Epochs: 250
- Dropout Rate: 30%
- Temperature for Text Generation: 0.8

## Results

- **Final Training Accuracy**: 79.86%
- **Final Training Loss**: 0.6725
- **Generated Text Example**:
  "The great financier was perfectly right in the afternoon Poirot gave a policeman getting of his own flesh and cry the nephew."

The model, tokenizer, and best weights have been saved for future testing and fine-tuning.

## Alternative Approaches

While the LSTM-based approach was selected, other methods considered include:
- GRU-based RNN
- Transformer-based models (GPT-2, BERT)
- Hybrid models combining CNN and LSTM
- Pretrained language models like GPT-3/4

## Performance Enhancement Techniques

To improve model performance, the following techniques could be applied:
1. Use Transformer-based models like GPT-2 for better handling of long-range dependencies.
2. Increase the size of the training dataset by using more books by Agatha Christie.
3. Hyperparameter optimization using grid search or Bayesian optimization.
4. Augment the training data with sentence paraphrasing.
5. Increase GloVe embedding size to 300D for improved word representation.

## Conclusion

The project successfully developed an LSTM-based language model capable of generating text in the style of Agatha Christie. Although LSTMs performed well, transformer models like GPT-2 are likely to produce more coherent text by efficiently handling long-range dependencies.

The dataset can be downloaded here in a single text file:
https://www.kaggle.com/datasets/jannesklaas/scifi-stories-text-corpus
