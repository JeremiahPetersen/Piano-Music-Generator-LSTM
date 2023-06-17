Piano Music Generator using LSTM.
This repository contains code for generating piano music using an LSTM neural network model. The code utilizes MIDI files as input data and implements functions for data preprocessing, model creation, training, and music generation.

Installation
To run the code, you need to have the following libraries installed:

os
numpy
pretty_midi
fluidsynth
sklearn
tensorflow

You can install these libraries using the following command:

pip install numpy pretty_midi fluidsynth scikit-learn tensorflow

Usage
Set the folder path containing the MIDI files you want to use for training. The MIDI files will be converted into NoteSequences using the convert_midi_to_notesequences function.

Preprocess the NoteSequences by converting them into binary piano rolls and splitting them into input/output sequences. You can specify the sequence length using the preprocess_notesequences function.

Create an LSTM model using the create_lstm_model function. Specify the input shape of the model.

Train the LSTM model using the train_lstm_model function. Provide the input and output sequences, along with any desired hyperparameters.

Generate new piano music using the trained LSTM model and a seed sequence. Use the generate_music function, specifying the model, seed sequence, and hyperparameters.

Save the generated music as a MIDI file using the save_midi_file function. Provide the piano roll, output file path, and sampling rate.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgments
The code in this repository is inspired by the work and examples provided in the TensorFlow and Keras documentation.

How the Script Works

1. The script begins by importing the necessary libraries and dependencies.
2. It defines functions for converting MIDI files to NoteSequences, preprocessing the NoteSequences, creating and training an LSTM model, generating new piano music, and saving the generated music as a MIDI file.
3. The convert_midi_to_notesequences function reads MIDI files from a specified folder and converts them into NoteSequences using the pretty_midi library.
4. The preprocess_notesequences function takes the NoteSequences and preprocesses them by converting them into binary piano rolls and splitting them into input/output sequences.
5. The create_lstm_model function defines the architecture of the LSTM model using the Sequential API from Keras. It includes LSTM layers, dropout layers for preventing overfitting, and a dense output layer with sigmoid activation.
6. The train_lstm_model function creates and trains the LSTM model using the input and output sequences. It compiles the model with binary cross-entropy loss and Adam optimizer, and includes callbacks for early stopping to prevent overfitting.
7. The generate_music function takes a trained LSTM model, a seed sequence, and generates new piano music by predicting the next notes based on the seed sequence.
8. The save_midi_file function saves the generated piano music as a MIDI file using the pretty_midi library.
9. In the main execution steps, the script sets the folder path of the piano MIDI files, converts them to NoteSequences, preprocesses the sequences, and splits them into training and testing sets.
10. If a trained model file exists, it loads the model; otherwise, it trains a new LSTM model using the training data and saves it.
11. It randomly selects a seed sequence from the testing set, generates new piano music using the trained model, and saves it as a MIDI file.
