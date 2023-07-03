from flask import Flask, render_template, request, redirect, url_for
import csv
from autocorrect import Speller
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import random
import nltk
from nltk.corpus import cmudict
from tensorflow.keras.utils import pad_sequences
from sklearn.preprocessing import StandardScaler
import textblob
import enchant
from enchant.checker import SpellChecker
from spellchecker import SpellChecker

app = Flask(__name__)
# Load the model and tokenizer
model = tf.keras.models.load_model('model_SongAI_save.tf')
with open('tokenizerSongAI.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
# Load the Standard Scaler
with open('scalerSongAI.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the maximum sequence length
max_sequence_length = 100

# Function to generate merged text
def generate_merged_text(seed_text, num_of_words, diversity, artist, genre, year):
    seed_tokens = tokenizer.texts_to_sequences([seed_text + artist + genre + year])
    seed_tokens = pad_sequences(seed_tokens, maxlen=max_sequence_length)

    numerical_inputs = scaler.transform([[year, num_of_words]])

    generated_text = seed_text

    for _ in range(num_of_words):
        prediction = model.predict([seed_tokens, numerical_inputs])
        prediction = prediction[0][-1]

        prediction /= np.sum(prediction)  # Normalize probabilities

        valid_indices = [idx for word, idx in tokenizer.word_index.items() if idx < len(prediction)]
        output_word_index = np.random.choice(valid_indices, p=prediction[valid_indices] / np.sum(prediction[valid_indices]))
        output_word = tokenizer.index_word.get(output_word_index, '')

        generated_text += ' ' + output_word

        seed_tokens = np.append(seed_tokens[:, 1:], [[output_word_index]], axis=1)

    return generated_text

# Function to correct spelling
def correct_spelling(paragraph):
    # Initialize the spell checker
    spell = Speller(lang='en')

    # Use textblob for initial spell checking
    blob = textblob.TextBlob(paragraph)
    corrected_paragraph = blob.correct()

    checked_paragraph = []
    for word in corrected_paragraph.words:
        # Check if the word is spelled correctly
        corrected_word = spell(word)
        checked_paragraph.append(corrected_word)

    return " ".join(checked_paragraph)

# Function to generate lyrics using Markov chain
def generate_lyrics(input_lyrics):
    # Create a list of words from the input lyrics
    words = input_lyrics.split(' ')

    # Create a Markov chain model
    markov_model = {}
    for i in range(1, len(words)):
        if words[i-1] not in markov_model:
            # If the word is not already in the model, add it
            markov_model[words[i-1]] = [words[i]]
        else:
            # If the word is already in the model, append the following word to the list
            markov_model[words[i-1]].append(words[i])

    # Choose a random word from the input lyrics to start the new lyrics
    current_word = random.choice(list(markov_model.keys()))
    new_lyrics = current_word.capitalize()

    # Generate the lyrics
    for i in range(len(words)-1):  # Subtract 1 because we already added the first word
        if current_word not in markov_model:
            break
        next_word = random.choice(markov_model[current_word])
        new_lyrics += ' ' + next_word
        current_word = next_word

    return new_lyrics

# Function to capitalize lines in the lyrics
def capitalize_lines(text):
    lines = text.split("\n")
    capitalized_lines = [line.capitalize() for line in lines]
    return "\n".join(capitalized_lines)

# Function to generate a verse
def generate_verse(lyrics):
    verse = []
    verse.append(lyrics)
    verse.append("")
    return "\n".join(verse)

# Function to generate a chorus
def generate_chorus(lyrics):
    chorus = []
    chorus.append(lyrics)
    chorus.append("")
    return "\n".join(chorus)

# Function to generate a bridge
def generate_bridge(lyrics):
    bridge = []
    bridge.append(lyrics)
    bridge.append("")
    return "\n".join(bridge)

# Function to generate a song based on user inputs
def generate_song(words, rhyme_scheme, verse_length, chorus_length, bridge_length):
    song_parts = []
    lyrics_idx = 0
    rhyme_idx = 0

    while lyrics_idx < len(words):
        if lyrics_idx + verse_length <= len(words) and rhyme_scheme[rhyme_idx] == 'A':
            verse_lyrics = " ".join(words[lyrics_idx:lyrics_idx + verse_length])
            song_parts.append(generate_verse(verse_lyrics))
            lyrics_idx += verse_length
        elif lyrics_idx + chorus_length <= len(words) and rhyme_scheme[rhyme_idx] == 'B':
            chorus_lyrics = " ".join(words[lyrics_idx:lyrics_idx + chorus_length])
            song_parts.append(generate_chorus(chorus_lyrics))
            lyrics_idx += chorus_length
        elif lyrics_idx + bridge_length <= len(words) and rhyme_scheme[rhyme_idx] == 'C' and bridge_length > 0:
            bridge_lyrics = " ".join(words[lyrics_idx:lyrics_idx + bridge_length])
            song_parts.append(generate_bridge(bridge_lyrics))
            lyrics_idx += bridge_length
        else:
            break  # add a break condition in case we can't increment lyrics_idx

        rhyme_idx = (rhyme_idx + 1) % len(rhyme_scheme)

    song = "\n".join(song_parts)
    return song


def check_credentials(username, password):
    with open('credentials.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if row[0] == username and row[1] == password:
                return True
    return False

@app.route('/')
def landing_page():
    return render_template('landing_page.html')

@app.route('/dashboard')
def content():
    return render_template('dashboard.html')

@app.route('/Song')
def home():
    return render_template('Song.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if check_credentials(username, password):
            return redirect(url_for('content'))
        else:
            error = 'Invalid username or password.'
    return render_template('signin.html', error=error)

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'POST':
        # Retrieve the form data and generate the song
        seed_text = request.form['seed_text']
        num_of_words = int(request.form['num_of_words'])
        diversity = float(request.form['diversity'])
        artist = request.form['artist']
        genre = request.form['genre']
        year = request.form['year']
        verse_length = int(request.form['verse_length'])
        chorus_length = int(request.form['chorus_length'])
        bridge_length = int(request.form['bridge_length'])
        rhyme_scheme = request.form['rhyme_scheme']

        generated_music = generate_merged_text(seed_text, num_of_words, diversity, artist, genre, year)
        corrected_paragraph = correct_spelling(generated_music)
        markov_song = capitalize_lines(generate_lyrics(corrected_paragraph))
        generated_song = generate_song(markov_song.split(), rhyme_scheme, verse_length, chorus_length, bridge_length)

        return render_template('result.html', generated_song=generated_song, generated_text=generated_music, new_lyrics=markov_song)
    else:
        # Render the initial form for generating song
        return render_template('Song.html')

@app.route('/generate-markov', methods=['POST'])
def generate_markov():
    generated_music = request.form['new_lyrics']
    markov_song = capitalize_lines(generate_lyrics(generated_music))
    return render_template('result.html', generated_text=generated_music, new_lyrics=markov_song)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
