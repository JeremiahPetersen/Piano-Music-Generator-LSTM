import os
import numpy as np
import pretty_midi
import fluidsynth
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

# Convert MIDI files to NoteSequences
def convert_midi_to_notesequences(folder):
    note_sequences = []
    for filename in os.listdir(folder):
        if filename.endswith(".mid"):
            try:
                pm = pretty_midi.PrettyMIDI(os.path.join(folder, filename))
                note_sequences.append(pm)
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")
    return note_sequences

# Preprocess the NoteSequences
def preprocess_notesequences(note_sequences, sequence_length=100):
    input_sequences, output_sequences = [], []
    for pm in note_sequences:
        piano_roll = pm.get_piano_roll(fs=10)
        piano_roll = (piano_roll > 0).astype(int)
        for i in range(0, piano_roll.shape[1] - sequence_length):
            input_sequences.append(piano_roll[:, i:i+sequence_length])
            output_sequences.append(piano_roll[:, i+1:i+sequence_length+1])
    return np.array(input_sequences), np.array(output_sequences)

# Create and train the LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))  # Reduced the number of neurons to 64
    model.add(Dropout(0.5))  # Increased dropout rate to 0.5
    model.add(LSTM(64, return_sequences=True))  # Reduced the number of neurons to 64
    model.add(Dropout(0.5))  # Increased dropout rate to 0.5
    model.add(Dense(input_shape[-1]))
    model.add(Activation('sigmoid'))
    return model

def train_lstm_model(input_sequences, output_sequences, num_epochs=100, batch_size=64):
    model = create_lstm_model(input_sequences.shape[1:])
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    # Added early stopping to stop training when the model starts overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(input_sequences, output_sequences, 
              epochs=num_epochs, 
              batch_size=batch_size,
              validation_split=0.2,  # Added a validation split for early stopping
              callbacks=[early_stopping])  # Added early stopping callback
    return model

# Generate new piano music
def generate_music(model, seed_sequence, num_steps=1024, threshold=0.5):
    generated_sequence = [seed_sequence]
    for _ in range(num_steps):
        prediction = model.predict(np.array([generated_sequence[-1]]))
        prediction[prediction >= threshold] = 1
        prediction[prediction < threshold] = 0
        generated_sequence.append(prediction[0])
    return np.concatenate(generated_sequence, axis=1)

# Save the generated music as a MIDI file
def save_midi_file(piano_roll, output_file, fs=10):
    pm = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    piano.notes = piano_roll_to_notes(piano_roll, fs=fs)
    pm.instruments.append(piano)
    pm.write(output_file)
    
# Define Piano Roll to Notes
def piano_roll_to_notes(piano_roll, fs=10):
    notes = []
    velocity = 100
    for pitch, pitch_roll in enumerate(piano_roll):
        onsets = np.where(np.diff(pitch_roll) == 1)[0]
        offsets = np.where(np.diff(pitch_roll) == -1)[0]

        if len(onsets) == 0 or len(offsets) == 0:
            continue

        if onsets[0] > offsets[0]:
            offsets = offsets[1:]

        if onsets[-1] >= offsets[-1]:
            onsets = onsets[:-1]

        for start, end in zip(onsets, offsets):
            start_time = start / fs
            end_time = end / fs
            note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=end_time)
            notes.append(note)

    return notes

# Main execution
folder = "YOUR FOLDER PATH OF PIANO MIDI FILES GOES HERE"
note_sequences = convert_midi_to_notesequences(folder)
input_sequences, output_sequences = preprocess_notesequences(note_sequences)

X_train, X_test, y_train, y_test = train_test_split(input_sequences, output_sequences, test_size=0.2, random_state=42)

model_file = "model.h5"

if os.path.exists(model_file):
    trained_model = tf.keras.models.load_model(model_file)
else:
    trained_model = train_lstm_model(X_train, y_train)
    trained_model.save(model_file)

seed_sequence = X_test[np.random.randint(0, len(X_test))]
generated_piano_roll = generate_music(trained_model, seed_sequence)

output_file = "PIANO_TRACK_1.mid"
save_midi_file(generated_piano_roll, output_file)
