# Piano-Music-Generator-LSTM
This repository contains code for generating music using an LSTM neural network model. The code utilizes MIDI files as input data and implements various functions for data preprocessing, model creation, training, and music generation.

Installation
To run the code, you need to have the following libraries installed:
os
numpy
pretty_midi
fluidsynth
sklearn
tensorflow
You can install these libraries using the following command:

shell
Copy code
pip install numpy pretty_midi fluidsynth scikit-learn tensorflow
Usage
Set the folder path containing the MIDI files you want to use for training. The MIDI files will be converted into NoteSequences using the convert_midi_to_notesequences function.

Preprocess the NoteSequences by converting them into binary piano rolls and splitting them into input/output sequences. You can specify the sequence length using the preprocess_notesequences function.

Create an LSTM model using the create_lstm_model function. Specify the input shape of the model.

Train the LSTM model using the train_lstm_model function. Provide the input and output sequences, along with any desired hyperparameters.

Generate new piano music using the trained LSTM model and a seed sequence. Use the generate_music function, specifying the model, seed sequence, and hyperparameters.

Save the generated music as a MIDI file using the save_midi_file function. Provide the piano roll, output file path, and sampling rate.

Example
python
Copy code
# Set the folder path containing MIDI files
folder_path = 'midi_files/'

# Convert MIDI files to NoteSequences
note_sequences = convert_midi_to_notesequences(folder_path)

# Preprocess NoteSequences
sequence_length = 100
input_sequences, output_sequences = preprocess_notesequences(note_sequences, sequence_length)

# Create and train the LSTM model
input_shape = input_sequences.shape[1:]
model = create_lstm_model(input_shape)
trained_model = train_lstm_model(input_sequences, output_sequences, model)

# Generate new music
seed_sequence = input_sequences[0]
generated_music = generate_music(trained_model, seed_sequence)

# Save the generated music as a MIDI file
output_file = 'generated_music.mid'
sampling_rate = 16
save_midi_file(generated_music, output_file, sampling_rate)
License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgments
The code in this repository is inspired by the work and examples provided in the TensorFlow and Keras documentation.
