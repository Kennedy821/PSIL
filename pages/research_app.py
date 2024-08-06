# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:25:27 2024

@author: worldcontroller
"""

import streamlit as st
import os
import zipfile
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
# from pydub import AudioSegment
# from pytube import YouTube, Playlist
from pathlib import Path
import librosa
import librosa.display
import tempfile
from zipfile import ZipFile
from PIL import Image
from io import BytesIO
import pandas as pd
import time
import soundfile as sf
import polars as pl
import concurrent.futures
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.datasets import make_blobs
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.metrics import silhouette_score
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.models import load_model
# from keras.applications.vgg16 import VGG16
# from keras.applications.vgg19 import VGG19
# from keras.models import Model
from joblib import dump, load
# from river import cluster, stream
import seaborn as sns
import gc
import random
from PIL import Image, ImageDraw, ImageSequence, ImageFont
import textwrap
from google.cloud import storage
from io import StringIO


from google.oauth2 import service_account
from google.cloud import storage
from st_files_connection import FilesConnection
import gcsfs
import yt_dlp
import timm
import torch
from PIL import Image
from torchvision import transforms
import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def preprocess_input(x):
    """
    Preprocesses a numpy array encoding a batch of images for EfficientNet.

    Args:
    - x (numpy.ndarray): Input image array to preprocess.

    Returns:
    - numpy.ndarray: Preprocessed image array.
    """
    # Ensure the input is a numpy array
    x = np.array(x, dtype=np.float32)
    
    # Scale pixel values to range [-1, 1]
    x /= 127.5
    x -= 1.0
    
    return x

def load_and_preprocess_image(image_path, target_size=(512, 512)):
    # Load the image
    img = Image.open(image_path)
    # Convert to RGB (if not already in RGB)
    img = img.convert('RGB')
    # Resize the image
    img = img.resize(target_size)
    
    # Convert the image to a NumPy array
    img_array = np.array(img)
    
    # Normalize the image
    img_array = img_array / 255.0
    
    return img_array

conn = st.connection('gcs', type=FilesConnection)


# Now you can use `index` for searching, etc.
     


def stream_data(word_to_stream):
    for word in word_to_stream.split(" "):
        yield word + " "
        time.sleep(0.25)

# Load the index as memory-mapped
#faiss_vector_db_path = "faiss_quantised_testing_100_clusters.index"

#index = faiss.read_index(faiss_vector_db_path)

def get_top_n_recommendations_gcs_version(list_of_song_components, list_of_features, n):
    filenames_ = list_of_song_components
    feat_ = list_of_features 
    #st.write(f"shape of uploaded array is: {feat.shape}")
    song_components_list = []
    song_components_recommendations_list = []

    start_time = time.time()
    while True:
        #download the indices from gcs
        blob = bucket.blob("queried_indices.csv")
        if blob.exists():

            # Download the file to a destination
            blob.download_to_filename(temp_dir+"queried_indices.csv")
            downloaded_indices_df = pd.read_csv(temp_dir+"queried_indices.csv")
            st.dataframe(downloaded_indices_df)
            break
        else:
            time.sleep(10)
    end_time = time.time()
    #st.write(f"Downloaded indices in {end_time - start_time} seconds")


    total_components_df = database_song_names_df.copy()
    total_components_df["target_song"] = total_components_df["song_name"].str.split("_spect").str[0]
    # total_components_df
    total_components_df["total_components"] = 1
    total_components_df = total_components_df[["target_song","total_components"]].groupby(["target_song"]).sum().sort_values("total_components", ascending=False).reset_index()
    

    for i in range(len(filenames_)):
        song= filenames_[i]
        song_components_list.append(song)


    
    for i in range(len(downloaded_indices_df.columns)):

        # I = [x for x in downloaded_indices_df.iloc[:,i].values]

        # total_components_df = database_song_names_df.copy()
        # total_components_df["target_song"] = total_components_df["song_name"].str.split("_spect").str[0]
        # # total_components_df
        # total_components_df["total_components"] = 1
        # total_components_df = total_components_df[["target_song","total_components"]].groupby(["target_song"]).sum().sort_values("total_components", ascending=False).reset_index()
        
        I = downloaded_indices_df.iloc[:, i].values
    
        results_df = database_song_names_df[database_song_names_df.index.isin(I)]
        results_df["origin_song_component"] = song
        results_df["origin_song"] = results_df["origin_song_component"].str.split("_spect").str[0]
        results_df["target_song"] = results_df["song_name"].str.split("_spect").str[0]
        results_df["counter"] =1



        pivoted_df = results_df[["origin_song","target_song","counter"]].groupby(["origin_song","target_song"]).sum().sort_values("counter", ascending=False).reset_index()
        pivoted_df = pivoted_df[pivoted_df.origin_song!=pivoted_df.target_song]
        pivoted_df = pivoted_df.merge(total_components_df, on="target_song")
        song_components_recommendations_list.append(pivoted_df)


    # st.markdown("total components df:")
    # st.dataframe(total_components_df)

    # st.markdown(f"results df: {len(results_df)}")
    # st.dataframe(results_df.head(40))

    # st.markdown("pivoted df:")
    # st.dataframe(pivoted_df)

    recommended_df = pd.concat(song_components_recommendations_list)

    st.markdown("recommended df:")
    st.dataframe(recommended_df)

    # as a song may appear multiple times the more similar it is we want to not double count the total components
    # the initial code below didn't account for this and therefore meant the sing similarity percentage was artificially low for some songs
    # recommended_df = recommended_df.groupby(["origin_song","target_song"]).sum().sort_values("counter", ascending=False).reset_index()
    # the version below ensures that the total components value that is kept is the true value
    recommended_df = pl.from_pandas(recommended_df).groupby(["origin_song","target_song"]).agg(pl.col("counter").sum(),
                                                                                               pl.col("total_components").median()).to_pandas().sort_values("counter", ascending=False).reset_index()

    recommended_df["origin_song_counter"] = total_uploaded_files
    recommended_df["uploaded_song_components"] = recommended_df["total_components"] / recommended_df["origin_song_counter"]
    recommended_df.loc[(recommended_df["uploaded_song_components"]<2) & (recommended_df["uploaded_song_components"]>0.5),"appropriate_song"] = "true"
    recommended_df.loc[recommended_df.appropriate_song.isna(),"appropriate_song"] = "false"



 

    recommended_df = recommended_df[(recommended_df.appropriate_song=="true")
                                    &(recommended_df.counter<=recommended_df.total_components)].reset_index().drop(columns="index")

    # this is a check to determine if the core index has sufficient rate of clustering
    # ideally this should at least be close to 50% across all joined songs
    match_rate = recommended_df.counter.sum() / recommended_df.total_components.sum()
    st.markdown(f"the match rate is: {round(match_rate*100,0)} %" )

    recommended_df["pct_similiar"] = recommended_df["counter"] / recommended_df["total_components"]
    recommended_df["ls_distance"] = recommended_df["uploaded_song_components"]*recommended_df["pct_similiar"]
    recommended_df = recommended_df.rename(columns={"target_song":"song_name"})
    recommended_df = recommended_df.sort_values("pct_similiar", ascending=False).head(n)
    recommended_df
    return recommended_df

def get_top_n_recommendations(list_of_song_components, list_of_features, n):
    filenames_ = list_of_song_components
    feat_ = list_of_features 
    st.write(f"shape of uploaded array is: {feat.shape}")
    song_components_list = []
    song_components_recommendations_list = []
    for i in range(len(filenames_)):
        song= filenames_[i]
        song_components_list.append(song)
        
        query_vector = feat_[i]
        query_vector = query_vector.reshape(1,query_vector.shape[0])
        
        k = 50                          # we want to see 4 nearest neighbors
        D, I = index.search(query_vector, k)     # actual search
    
        
        total_components_df = database_song_names_df.copy()
        total_components_df["target_song"] = total_components_df["song_name"].str.split("_spect").str[0]
        total_components_df["total_components"] = 1
        total_components_df = total_components_df[["target_song","total_components"]].groupby(["target_song"]).sum().sort_values("total_components", ascending=False).reset_index()
        
    
        results_df = database_song_names_df[database_song_names_df.index.isin(I[0])]
        results_df["origin_song_component"] = song
        results_df["origin_song"] = results_df["origin_song_component"].str.split("_spect").str[0]
        results_df["target_song"] = results_df["song_name"].str.split("_spect").str[0]
        results_df["counter"] =1
        pivoted_df = results_df[["origin_song","target_song","counter"]].groupby(["origin_song","target_song"]).sum().sort_values("counter", ascending=False).reset_index()
        pivoted_df = pivoted_df[pivoted_df.origin_song!=pivoted_df.target_song]
        pivoted_df = pivoted_df.merge(total_components_df, on="target_song")
        song_components_recommendations_list.append(pivoted_df)
        
        
    recommended_df = pd.concat(song_components_recommendations_list)
    recommended_df = recommended_df.groupby(["origin_song","target_song"]).sum().sort_values("counter", ascending=False).reset_index()
    recommended_df["pct_similiar"] = recommended_df["counter"] / recommended_df["total_components"]
    recommended_df["ls_distance"] = recommended_df["counter"]*recommended_df["pct_similiar"]
    recommended_df = recommended_df.rename(columns={"target_song":"song_name"})
    recommended_df = recommended_df.sort_values("pct_similiar", ascending=False).head(n)
    #recommended_df
    return recommended_df

def reshape_features_to_pairs(features):
    # Assuming the features have an even length; otherwise, you may need to handle the last element
    return [list(features[i:i+2]) for i in range(0, len(features), 2)]

def convert_image_path_to_target_list_of_lists(image_path):
    features = extract_features(image_path)
    print(features.shape)
    output = [reshape_features_to_pairs(x) for x in features.tolist()]
    return output

def get_clusters_vectorised(feature_vector):
    cluster_list = []
    for n, _ in stream.iter_array(feature_vector):
        #print(n,'|',_)

        predicted_cluster = loaded_denstream.predict_one(n)
        loaded_denstream.learn_one(n)
        cluster_list.append(predicted_cluster)
    return cluster_list

def get_clusters_vectorised_prod(feature_vector):
    cluster_list = []
    for n, _ in stream.iter_array(feature_vector):
        #print(n,'|',_)

        predicted_cluster = loaded_denstream.predict_one(n)
        cluster_list.append(predicted_cluster)
    return cluster_list

def get_clusters_from_denstream(dataframe, string_w_extracted_feature_vector):
    counter = 0
    dataframe = dataframe.set_index(dataframe.columns[0])

    cluster_list = dataframe[string_w_extracted_feature_vector].apply(lambda x: get_clusters_vectorised(x))

    df = pd.DataFrame(cluster_list)
    df = df[df.columns[0]].apply(pd.Series)
    df = df.reset_index()
    df = df.melt(id_vars=df.columns[0]).drop(columns="variable")
    df.columns = ['song_component','cluster']
    df.loc[df.song_component.isin(all_files), 'chosen_flag'] = 1
    df.loc[df.chosen_flag.isna(),'chosen_flag'] = 0
    df["song_name"] = df["song_component"].str.split("spect").str[0]
    df["song_name"] = df["song_name"].str[:-1]

    st.write(f"chosen song is: {chosen_song_name}")
    df["counter"] = 1
    pivoted_df_x = df.groupby(["song_name","cluster"]).sum().reset_index().pivot(index="song_name", columns = "cluster", values="counter").fillna(0).reset_index()
    pivoted_df_x= pivoted_df_x.set_index("song_name")
    return pivoted_df_x

#@st.cache_resource()  # Cache the function to avoid reprocessing unchanged input
def generate_frames_with_text(input_gif_path, full_text):
    with Image.open(input_gif_path) as im:
        frames = []
        center_x, center_y = (im.size[0] // 2), im.size[1] // 2
        approx_char_width = 6  # Approximate pixel width of a character

        # Split text into lines, treating each line as a separate chunk of text to display
        lines = full_text.split("\n")

        # Generate a sequence of frames with each line added progressively
        current_text = []
        for line in lines:
            current_text.append(line)  # Add the line to the current text
            temp_frames = []
            for frame in ImageSequence.Iterator(im):
                frame = frame.convert("RGBA")
                draw = ImageDraw.Draw(frame)

                # Calculate positions and draw each line of current_text
                for i, text_line in enumerate(current_text):
                    text_width = len(text_line) * approx_char_width
                    text_x = center_x - text_width // 2
                    text_y = center_y - 10 + 15 * (i - len(current_text) // 2)
                    draw.text((text_x, text_y), text_line, fill="black")
                
                temp_frames.append(frame.copy())

            # Store BytesIO for each stage of text
            byte_io = BytesIO()
            temp_frames[0].save(byte_io, format='GIF', save_all=True, append_images=temp_frames[1:], optimize=False, loop=0)
            byte_io.seek(0)
            frames.append(byte_io)

        return frames


    
def display_animated_text(input_gif_path, input_text):
    input_gif_path = input_gif_path  # Update with your GIF path
    full_text = input_text  # The text to animate
    font_path = 'Arial.ttf'  # Provide the path to your TTF font
    font_size = 10  # Set your desired font size
    font_size = 20
    max_width = 20  # Max width in characters, adjust as needed
    frames = generate_frames_with_text(input_gif_path, full_text)
    placeholder = st.empty()
    
    # Loop to display each frame
    for frame in frames:
        placeholder.image(frame, use_column_width=True)
        time.sleep(0.25)  # Adjust timing according to your preference

     
def add_text_to_gif(input_gif_path, full_text, update_interval=0.1):
    # Open the original GIF
    with Image.open(input_gif_path) as im:
        # Define a draw object placeholder outside of the loop
        d = ImageDraw.Draw(im)
        
        # Calculate text size using textbbox for the full text
        text_bbox = d.textbbox((0, 0), full_text, font=d.getfont())
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_position = ((im.size[0] - text_width) // 2, (im.size[1] - text_height) // 2)
        
        # Iterate over the text to create an animation of streaming text
        for i in range(len(full_text) + 1):
            # List to hold the processed frames for each stage of the text
            frames = []
            current_text = full_text[:i]
            
            # Iterate over each frame in the animated GIF
            for frame in ImageSequence.Iterator(im):
                # Convert the frame to RGBA
                frame = frame.convert("RGBA")
                
                # Draw the text on the frame
                d = ImageDraw.Draw(frame)
                d.text(text_position, current_text, fill="black")
                
                # Append the edited frame to the list of frames
                frames.append(frame)

            # Create a new GIF-like object in memory for each stage of the text
            byte_io = BytesIO()
            frames[0].save(byte_io, format='GIF', save_all=True, append_images=frames[1:], optimize=False, duration=im.info['duration'], loop=0)
            byte_io.seek(0)

            # Yield the BytesIO object to be displayed
            yield byte_io
            time.sleep(update_interval)


#gif_path = "square_processed_image.gif"




# Hypothetical function to convert an image back to audio
def image_to_audio(image):
    # Implementation of inverse spectogram
    # This is a placeholder and needs to be replaced with actual logic
    pass

def base64_to_image(base64_string):
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    return img

def reconstruct_audio_from_spectogram(S_mag, sr=22050, hop_length=512, n_iter=32):
    """
    Reconstruct an audio signal from a magnitude spectogram using the Griffin-Lim algorithm.

    :param S_mag: The magnitude of the spectogram (e.g., as a numpy array).
    :param sr: The sample rate of the audio signal.
    :param hop_length: The hop length used for the STFT.
    :param n_iter: The number of iterations for the Griffin-Lim algorithm.
    :return: The reconstructed audio time series.
    """
    # Use Griffin-Lim algorithm to reconstruct the phase
    S_complex = librosa.magphase(librosa.stft(S_mag, hop_length=hop_length))[0]
    y_reconstructed = librosa.griffinlim(S_complex, hop_length=hop_length, n_iter=n_iter)

    return y_reconstructed
def generate_and_display_spectrogram(segment, sr, temp_dir, segment_index, song_name):
    plt.figure(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=segment, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    # plt.colorbar(format='%+2.0f dB')
    plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Minimize padding

    
    # Save the spectrogram as a PNG in the temporary directory
    spectrogram_path = os.path.join(temp_dir, f'{song_name}_spectrogram_{segment_index+1}.png')
    plt.savefig(spectrogram_path)
    plt.close()

    # Display the spectrogram in the Streamlit app
    image = Image.open(spectrogram_path)
    st.image(image, caption=f'{song_name} Segment {segment_index+1}', use_column_width=True)

    return spectrogram_path

def generate_song_spectrogram(song_link):
    all_spectrograms = []
    
    yt = YouTube(song_link)
    if yt.length / 60 < 10:  # Process only videos shorter than 10 minutes
        audio_stream = yt.streams.filter(only_audio=True).first()
        with tempfile.TemporaryDirectory() as temp_song_dir:
            audio_path = audio_stream.download(output_path=temp_song_dir)
            y, sr = librosa.load(audio_path, sr=None)
            segment_length_samples = 30 * sr
            num_segments = len(y) // segment_length_samples

            for index in range(num_segments):
                start_sample = index * segment_length_samples
                end_sample = start_sample + segment_length_samples
                segment = y[start_sample:end_sample]
                
                song_name = yt.title.replace("|", "").replace("/", "-")  # Ensure file name is OS-friendly
                spectrogram_path = generate_and_display_spectrogram(segment, sr, temp_song_dir, index, song_name)
                all_spectrograms.append(spectrogram_path)
    
    return all_spectrograms

def generate_spectrogram(song_link):
    st.markdown(song_link)
    all_spectrograms = []
    # song_file_directory = temp_dir

    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'{temp_dir}/%(title)s.%(ext)s',  # Save to 'downloads' folder with title as filename
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',  # You can also use 'wav'
                'preferredquality': '192',
            }],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(song_link, download=False)
            # st.markdown(info_dict)
            video_length_minutes = info_dict.get('duration') / 60
            # st.markdown(video_length_minutes)
    except Exception as e:
        st.markdown(f"{e} could not be processed.")
        video_length_minutes = 1000
    # Process the song if it's less than 10 minutes
    if video_length_minutes < 10:
        with tempfile.TemporaryDirectory() as temp_song_dir:
            try:
    
                song_file_directory = temp_song_dir


                # ydl_opts = {
                #     'format': 'bestaudio/best',
                #     'outtmpl': f'{song_file_directory}/%(title)s.%(ext)s',  # Save to 'downloads' folder with title as filename
                #     'postprocessors': [{
                #         'key': 'FFmpegExtractAudio',
                #         'preferredcodec': 'wav',  # You can also use 'wav'
                #         'preferredquality': '192',
                #     }],
                # }


                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': f'{song_file_directory}/%(title)s.%(ext)s',  # Save to 'downloads' folder with title as filename
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',  # You can also use 'wav'
                        'preferredquality': '192',
                    }],
                    'replace_in_metadata': [('title', r' \|\_', ''),
                                            # ('title', r'_', '_'),
                                            ('title', r'\"', '_'),
                                             ('title', r'\-', ''), ],  # Replace '|' with '-' in the title
                    'restrictfilenames': True,  # Optional: further sanitize filenames
                }

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # ydl.download([song_link])
                    info_dict = ydl.extract_info(song_link, download=True)
                    # mp3_song_name = info_dict.get('title', None).replace("\|","-")#.replace('.mp4','.mp3').replace('.Mp4','.mp3').replace('.MP4','.mp3').replace("|","")
                    mp3_song_name = re.sub(r'\|', '', info_dict.get('title', '')).replace(" ","_").replace("-","_").replace("__","_").replace('\"','')
                    video_length_minutes = info_dict.get('duration') / 60
                
                
                st.markdown(f"{mp3_song_name}:{video_length_minutes} mins")

                # Function to sanitize filenames
                def sanitize_filename(filename):
                    return filename.replace("|","")

                # Function to rename files in the directory
                def rename_files_in_directory(directory):
                    for filename in os.listdir(directory):
                        if '|' in filename:
                            sanitized_filename = sanitize_filename(filename)
                            original_path = os.path.join(directory, filename)
                            sanitized_path = os.path.join(directory, sanitized_filename)
                            os.rename(original_path, sanitized_path)
                            print(f'Renamed: {original_path} to {sanitized_path}')

                for i in os.listdir(song_file_directory):
                    st.markdown(i)
                    # st.markdown(i.replace(" ","_"))

                for n in os.listdir(song_file_directory)[:]:
                    # st.markdown(f"song name mp3: {mp3_song_name}")
                    # st.markdown(n)
                    ratio = fuzz.ratio(mp3_song_name, n)
                    st.markdown(ratio)

                    if ratio >0.8:
                        # st.markdown(n)
                        # song_file_directory = Path.cwd()
                        audio_path = os.path.join(song_file_directory, n)
        # =============================================================================
        #                 with open(audio_path, 'wb') as f:
        #                     f.write(audio_file.getvalue())
        # =============================================================================
            
                        y, sr = librosa.load(audio_path, sr=None)
                        segment_length_samples = 30 * sr
                        num_segments = len(y) // segment_length_samples
            
                        saved_paths = []
        
                        with st.expander(n):
                            st.audio(audio_path)
                            for i in range(num_segments):
                                
                                start_sample = i * segment_length_samples
                                end_sample = start_sample + segment_length_samples
                                segment = y[start_sample:end_sample]
                
                                # Generate, save, and display the spectrogram
                                spectrogram_path = generate_and_display_spectrogram(segment, sr, temp_dir, i, mp3_song_name)
                                saved_paths.append(spectrogram_path)
                                all_spectrograms.append(spectrogram_path)

                        return saved_paths
            except Exception as e:
                st.write(f"An error occurred when processing {mp3_song_name} {str(e)}")
                pass
def image_to_base64(pil_image):
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')  # Using PNG as a default format
    base64_string = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    return base64_string

def process_files(temp_dir):
    # Example function to process files in the temporary directory
    # Here you can implement your logic that requires posix.DirEntry objects
    for entry in os.scandir(temp_dir):
        # st.markdown(entry)
        if entry.is_file() and entry.name.endswith(('.png', '.jpg', '.jpeg')):
            #st.write(f"Processing file: {entry.name}")
            # Add your processing logic here   
            flowers.append(entry.name)

def extract_features(file, model):
    # load the image as a 224x224 array
    img = load_img(os.path.join(directory,file), target_size=(224,224))
    #st.markdown(img)
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx)
    return features

def extract_features_efficient_net(file, model,index_value):
    # load the image as a 224x224 array
    # img = load_img(os.path.join(directory,file), target_size=(512,512))
    img = load_and_preprocess_image(os.path.join(directory,file))
    img = np.array(img)
    reshaped_img = img.reshape(1,512,512,3)
    imgx = preprocess_input(reshaped_img)
    st.markdown(imgx)
    # Run the image through the model
    with torch.no_grad():
        features = eff_net_model(imgx)
    # prepare image for model
    # get the feature vector
    # features = eff_net_model.predict(imgx)
    print(index_value)
    data[index_value] = features

    del img
    del imgx
    del reshaped_img
    return features.astype(np.float32) 

# Define the image loading and preprocessing function
def load_and_preprocess_image(image_path, target_size=(512, 512)):
    """
    Loads an image from the specified file path and preprocesses it.

    Args:
    - image_path (str): Path to the image file.
    - target_size (tuple of int): Desired size (width, height) to resize the image.

    Returns:
    - torch.Tensor: Preprocessed image tensor.
    """
    # Load the image
    img = Image.open(image_path).convert('RGB')
    
    # Define preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),  # Convert the image to a tensor with values in [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])
    
    # Apply preprocessing
    img_tensor = preprocess(img)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor

# Define the feature extraction function
def extract_features_efficient_net(file, model, index_value):
    """
    Extracts features from an image using EfficientNet.

    Args:
    - file (str): Image file name.
    - model (torch.nn.Module): Pretrained EfficientNet model.
    - index_value (int): Index value for data storage.
    - directory (str): Directory containing the image file.

    Returns:
    - np.ndarray: Extracted features as a NumPy array.
    """
    # Load and preprocess the image
    img_tensor = load_and_preprocess_image(os.path.join(directory,file))
    
    # Run the image through the model
    with torch.no_grad():
        features = model(img_tensor)
    
    # Convert features to NumPy array
    features_np = features.numpy()
    
    # Store the features in the data dictionary
    data[index_value] = features_np
    
    # Return features as float32
    return features_np.astype(np.float32)  
# Set page title and favicon
#st.set_page_config(page_title="PSIL Basic", page_icon=":musical_note:")


st.title("PSIL: Research production version")

# Input interface
st.subheader("Input Songs")
song_link = st.text_input("Enter the YouTube link of the song or playlist:")
#generate_playlist = st.checkbox("Generate spectrograms for a playlist")

if st.button("Recommend me songs"):
    with st.spinner('Processing your file(s)...'):
        if song_link:
            with tempfile.TemporaryDirectory() as temp_dir:

                # generate the spectorgams
                spectograms_paths = generate_spectrogram(song_link)
# =============================================================================
                # st.markdown(spectograms_paths)
# =============================================================================
                st.success('Done!')
                
                # convert these spectograms to be processed by Efficient Net
                # Data processing section
                # Loading section
                names_of_songs_in_upload_order = []
                images_list = []
                flowers = []
                
                directory = temp_dir
                # List all files in the folder
                all_files = os.listdir(directory)
                #print(all_files)
                # Filter out files that are not .png, .jpg, or .jpeg
                image_files = [file for file in all_files if file.endswith(('.png', '.jpg', '.jpeg'))]
                
                
                total_uploaded_files = len(image_files)
                # st.markdown(total_uploaded_files)
                # st.markdown(len(names_of_songs_in_upload_order))
                # Display the progress bar
                if len(image_files)>1:
                    #progress = len(image_files)
                    #st.progress(progress / total_uploaded_files)  # Assuming a max of 500 files for full progress
                
# =============================================================================
#                     st.write(f"Total files loaded: {len(image_files)} / {total_uploaded_files}")
# =============================================================================
                    total_number_of_assumed_songs = len(list(set([x.split("spect")[0] for x in image_files])))
# =============================================================================
#                     st.write(f"It looks like you have loaded {total_number_of_assumed_songs} songs.")
# =============================================================================
                
                for image_file in image_files:
                    # Define the full path for the original file
                    original_file_path =  image_file
                    # Add the name of the file to an ordered list based on their name
                    names_of_songs_in_upload_order.append(image_file)
            
                process_files(directory)# Extract feature vectors from images
                chosen_song_name_df = pd.DataFrame(names_of_songs_in_upload_order)
                chosen_song_name_df.columns = ["song_component"]
                chosen_song_name_df["song_name"] = chosen_song_name_df.song_component.str.split("spect").str[0].str[:-1]
                # st.dataframe(chosen_song_name_df)
                chosen_song_name = chosen_song_name_df["song_name"].value_counts().index[0]
                #st.markdown(f"{chosen_song_name}")
                
                # model = VGG19()
                # model = Model(inputs = model.inputs, outputs = model.layers[-2].output)
                # import os
                # # os.environ['TF_METAL'] = '1'
                # from tensorflow.keras import mixed_precision

                # # Set mixed precision policy
                # policy = mixed_precision.Policy('mixed_float32')
                # mixed_precision.set_global_policy(policy)

                # from tensorflow.keras.applications import EfficientNetB0
                # from tensorflow.keras.applications.efficientnet import preprocess_input


                # st.write("the VGG Net model is being loaded")
                
                # Load EfficientNetB0 model
                # eff_net_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(512, 512, 3))

                # Load EfficientNetB0 model
                eff_net_model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0, global_pool='')
                # model.eval()

                
                # st.write("Success: the VGG Net model has been loaded")

                #chosen_model = "vgg"
                chosen_model = "EfficientNet" 
                

             
                
                    
                data = {}
                features_df = pd.DataFrame()
                
                
                # st.markdown(flowers)
                p = r"/Users/tariromashongamhende/Documents/ml_projects/psil/rebirth/vectors"
# =============================================================================
                # st.write("running image feature extraction with CNN")
# =============================================================================
                # loop through each image in the dataset
                
                # Create a progress bar instance
                #progress_bar = st.progress(0)
                image_df =  pd.DataFrame([flowers]).T
                image_df.columns = ["image_names"]
                counter = np.arange(0,len(image_df))
                image_df["counter"] = counter
                image_df["counter"] = image_df["counter"].astype(int)
# =============================================================================
                # st.write(image_df)
# =============================================================================
                # can this run on all of it?
                feat = image_df.apply(lambda x:extract_features_efficient_net(x.image_names,eff_net_model,x.counter),axis=1)


                # Number of steps in your process
                number_of_steps = len(flowers)
    
                filenames = np.array(list(data.keys()))
                
                # get a list of just the features
                init_feat = np.array(list(data.values()))
                del data
                
# =============================================================================
#                 st.write(init_feat.shape)
# =============================================================================
                
                # reshape output from efficient net and preprocess for ann
                start_time = time.time()

                # Create a memory-mapped file
                shape = (total_uploaded_files, 327680)  # Example shape
                dtype = np.float32   
                #memmap_file = np.memmap('data.dat', dtype=dtype, mode='w+', shape=shape)
                
                bag_of_arrays = []
# =============================================================================
#                 print("starting to reshape features")
# =============================================================================
                if chosen_model == "vgg":
                    
                    # reshape so that ther:e are 210 samples of 4096 vectors
                    feat = init_feat.reshape(-1,4096)
                
                    del init_feat                 
                
                    
                else:
                    # Reshape the features to a single vector per image in the batch
                    feat = init_feat.reshape(-1, 16 * 16 * 1280) 
                
                    del init_feat
# =============================================================================
                # st.write(f"this is the shape of the processed song: {feat.shape}")
# =============================================================================
                end_time = time.time()



                #st.write("trying to save file to google cloud")
                # Convert DataFrame to CSV string.
                #image_df.to_csv("gs://psil-app-backend/test_upload.csv")
                def get_storage_client():
                    # Construct the credentials dictionary from the secrets
                    creds_dict = {
                        "type": st.secrets["google_credentials"]["type"],
                        "project_id": st.secrets["google_credentials"]["project_id"],
                        "private_key_id": st.secrets["google_credentials"]["private_key_id"],
                        "private_key": st.secrets["google_credentials"]["private_key"],
                        "client_email": st.secrets["google_credentials"]["client_email"],
                        "client_id": st.secrets["google_credentials"]["client_id"],
                        "auth_uri": st.secrets["google_credentials"]["auth_uri"],
                        "token_uri": st.secrets["google_credentials"]["token_uri"],
                        "auth_provider_x509_cert_url": st.secrets["google_credentials"]["auth_provider_x509_cert_url"],
                        "client_x509_cert_url": st.secrets["google_credentials"]["client_x509_cert_url"]
                    }
                    # Create a client using the credentials dictionary
                    client = storage.Client.from_service_account_info(creds_dict)
                    return client

                path_to_private_key = 'utility-braid-351906-bde1a70eb39a.json'
                client = storage.Client.from_service_account_json(json_credentials_path=path_to_private_key)

                #client = get_storage_client()

                # The bucket on GCS in which to write the CSV file
                bucket = client.bucket('psil-app-backend-2')
                # The name assigned to the CSV file on GCS
                blob = bucket.blob('my_data.csv')
                blob.upload_from_string(image_df.to_csv(), 'text/csv')
                #st.write("successfully uploaded file to google cloud")


                #st.write("attempting to write np.memmap file to gcs")
                #st.markdown(type(feat))
                shape = (feat.shape[0], feat.shape[1])  # Example shape
                dtype = feat.dtype
                #st.write(f"shape of input song written to gcs: {feat.shape}")
                memmap_file = np.memmap(f'{temp_dir}/uploaded_song_feature_vector.dat', dtype=dtype, mode='w+', shape=shape)

                # Write the result array to the memory-mapped file
                memmap_file[:] = feat
                memmap_file.flush()  # Manually flush changes to disk
                gc.collect()  # Force garbage collection

                del memmap_file


                def upload_file_to_gcs(bucket_name, source_file_name, destination_blob_name):
                    """Uploads a file to the bucket."""
                    bucket = client.bucket(bucket_name)
                    blob = bucket.blob(destination_blob_name)

                    blob.upload_from_filename(source_file_name)
                time.sleep(2)
                # Example usage
                bucket_name = 'psil-app-backend-2'
                source_file_name = f'{temp_dir}/uploaded_song_feature_vector.dat'
                destination_blob_name  = 'uploaded_song_feature_vector.dat'  # Assuming this file has already been created

                # Upload the data
                upload_file_to_gcs(bucket_name, source_file_name, destination_blob_name)

                st.markdown(names_of_songs_in_upload_order)

                      
                #st.write("successfully uploaded vector file to google cloud")
                
                

# =============================================================================
#                 st.write("predictions done")
# =============================================================================
                #blob = bucket.blob("myfile.csv")
                # Download the file to a destination
                #blob.download_to_filename(temp_dir+"myfile.csv")
                #downloaded_df = pd.read_csv(temp_dir+"myfile.csv")

                #st.dataframe(downloaded_df)




# =============================================================================
#                 st.write(f"Reshaping took {end_time - start_time:.6f} seconds")
# =============================================================================
                # predict cluster for song
                
                # load son of psil umap reducers
                
                # Load the dimension reducer models from disk

                n_for_uploaded_image = len(filenames)
                #umap_main_path = Path("C:\\Users\\worldcontroller\\Documents\\PSIL\\rebirth\\son_of_psil\\1_main_pca_reducer.joblib")

                
                # load the names of the images in the memmap_array
                # song_names_df_path = Path("/Users/tariromashongamhende/Downloads/psil_crawler_full_names_of_songs_in_upload_order.parquet.gzip")


                
                
                song_names_df_path = "/Users/tariromashongamhende/Downloads/psil_crawler_song_names_mapped_to_latest_index.parquet.gzip"#.head(5000)
                saved_songs_names_df = pd.read_parquet(song_names_df_path, engine="pyarrow")#.head(5000)

                database_song_names_df = saved_songs_names_df


# =============================================================================
#                 st.markdown(saved_songs_names_df.head())
# =============================================================================
                old_song_names_in_order = [x for x in saved_songs_names_df.song_name.values]
                #names_of_songs_in_upload_order = old_song_names_in_order + names_of_songs_in_upload_order
                


                filtered_selection_n = 5
                
                # load the links reference table
                # playlist_links_path = '/Users/tariromashongamhende/Documents/ml_projects/psil/all_playlist_links/all_playlist_links_letters/'
                # df_container = []
                # for file in os.listdir(playlist_links_path):
                #     if "parquet" in file:
                #         int_df = pd.read_parquet(f"{playlist_links_path}{file}", engine="pyarrow")
                #         df_container.append(int_df)

                # links_df = pd.concat(df_container).reset_index().drop(columns="index")


                # master_links_filepath = Path("new_playlist_links_a_to_f.csv")
                # master_links_filepath = Path("new_june_master_playlist_links_rehoboam_links_master.csv")

                master_links_filepath = Path("new_playlist_links_a_to_z.csv")

                links_df = pd.read_csv(master_links_filepath)
                
                top_recommendations_df = get_top_n_recommendations_gcs_version(names_of_songs_in_upload_order, feat, 10)

                # top_recommendations_df

                #top_recommendations_df = get_top_n_recommendations(names_of_songs_in_upload_order, feat, 5)
                
                top_recommendations_links_df = top_recommendations_df.merge(links_df[["song_name", "song_links"]], on="song_name", how="left")[["song_name", "song_links"]].reset_index().drop(columns="index").drop_duplicates("song_name")
                st.dataframe(top_recommendations_links_df)
                
                # Create a dictionary mapping song names to links
                song_links_map = dict(zip(top_recommendations_links_df['song_name'], top_recommendations_links_df['song_links']))
                
                markdown_list_items = []
                markdown_list_items_no_links = []
                for song in top_recommendations_df['song_name']:
                    song_len = len(song)
                    if song_len > 30:
                        song_part_1 = ''.join(song.split(" ")[:5])
                        song_part_2 = ''.join(song.split(" ")[5:15])
                        song_part_3 = ''.join(song.split(" ")[15:])
                
                
                        # Check if the song has a corresponding link
                        link = song_links_map.get(song)
                        if pd.notna(link):  # Check if link is not NaN
                            # If a link exists, format it with a hyperlink icon
                            if len(song_part_1)>1 and len(song_part_2)>1 and len(song_part_3)>1:
                                markdown_list_items.append(f"- {song_part_1} \n {song_part_2} \n {song_part_3} [▶️]({link})\n")
                                markdown_list_items_no_links.append(f"- {song_part_1} \n {song_part_2}\n {song_part_3}\n")
                            elif len(song_part_1)>1 and len(song_part_2)>1 and len(song_part_3)<1:
                                markdown_list_items.append(f"- {song_part_1} \n {song_part_2} [▶️]({link})\n")
                                markdown_list_items_no_links.append(f"- {song_part_1} \n {song_part_2}\n")
                 
                        else:
                            # If no link exists, just add the song name
                            markdown_list_items.append(f"- {song_part_1} \n {song_part_2} \n")
                            markdown_list_items_no_links.append(f"- {song_part_1} \n {song_part_2} \n")
                    else:
                        # Check if the song has a corresponding link
                        link = song_links_map.get(song)
                        if pd.notna(link):  # Check if link is not NaN
                            # If a link exists, format it with a hyperlink icon
                            markdown_list_items.append(f"- {song} [▶️]({link})\n")
                            markdown_list_items_no_links.append(f"- {song}\n")
                
                        else:
                            # If no link exists, just add the song name
                            markdown_list_items.append(f"- {song}\n")
                            markdown_list_items_no_links.append(f"- {song}\n")
                
                # Join the list items into a single Markdown string
                markdown_list = "\n".join(markdown_list_items)
                markdown_list_no_links = "\n".join(markdown_list_items_no_links)
                
                # Display in Streamlit
                st.header("Here are your recommendations")
                
                #gif_with_text = display_animated_text(gif_path,markdown_list_no_links)
                
                st.write_stream(stream_data(markdown_list))

                
                with st.expander("See how much we think you'll like these based on your uploaded song"): 
                    starting_value = 0  # Your starting/reference value
                    values = top_recommendations_df.sort_values("ls_distance", ascending=True).ls_distance  # Individual values to compare
                    labels = [x for x in top_recommendations_df.sort_values("ls_distance", ascending=True).song_name.values]
                    #song_names_markdown_list = ""
                    #st.markdown(labels)
                
                    fig, ax = plt.subplots(figsize=(5, 10))
                
                    # Plotting each point with a line to the starting value
                    for i, value in enumerate(values):
                        ax.plot([starting_value, value],[labels[i], labels[i]], 'grey')  # Line
                        ax.plot(value, labels[i] , 'o', color='blue')  # Dot
                    
                    # Highlight the starting value across the chart
                    #ax.axvline(starting_value, color='red', linestyle='--', label='Chosen song')
                    
                    #plt.title('Proximity to Starting Value')
                    plt.xticks(fontsize=15, rotation=45)
                    plt.yticks(fontsize=20)
                    plt.legend()
                    sns.despine()
                    st.pyplot(fig)
                
                
            
            
            
            
        else:
            st.warning("Please enter the YouTube link of the song you'd like to get recommendations for.")