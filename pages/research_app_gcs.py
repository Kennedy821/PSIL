# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:25:27 2024

@author: worldcontroller
"""

import streamlit as st
# import os
# import zipfile
# import io
# import base64
import numpy as np
import matplotlib.pyplot as plt
# from pydub import AudioSegment
# from pytube import YouTube, Playlist
from pathlib import Path
# import librosa
# import librosa.display
import tempfile
# from zipfile import ZipFile
from PIL import Image
from io import BytesIO
import pandas as pd
import time
# import soundfile as sf
import polars as pl
# import concurrent.futures
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
# from joblib import dump, load
# from river import cluster, stream
import seaborn as sns
# import gc
# import random
from PIL import Image, ImageDraw, ImageSequence, ImageFont
# import textwrap
# from google.cloud import storage
# from io import StringIO


from google.oauth2 import service_account
from google.cloud import storage
from st_files_connection import FilesConnection
# import gcsfs
# import yt_dlp
# import timm
# import torch
from PIL import Image
# from torchvision import transforms
# import re
# from fuzzywuzzy import fuzz
# from fuzzywuzzy import process

im = Image.open('slug_logo.png')
st.set_page_config(
    page_title="PSIL",
    page_icon=im,
)

conn = st.connection('gcs', type=FilesConnection)


# Now you can use `index` for searching, etc.
     


def stream_data(word_to_stream):
    for word in word_to_stream.split(" "):
        yield word + " "
        time.sleep(0.25)

# Load the index as memory-mapped
#faiss_vector_db_path = "faiss_quantised_testing_100_clusters.index"

#index = faiss.read_index(faiss_vector_db_path)

def get_top_n_recommendations_gcs_version(n):
    while True:


        #download the indices from gcs
        blob = bucket.blob("my_data.csv")
        if blob.exists():

            # Download the file to a destination
            blob.download_to_filename(temp_dir+"my_data.csv")
            song_components_df = pd.read_csv(temp_dir+"my_data.csv")
            song_components_df = song_components_df[[song_components_df.columns[1]]]

            # st.dataframe(song_components_df)
            break
        else:
            time.sleep(5)


    filenames_ = [x for x in song_components_df.iloc[:,0].values]
    total_uploaded_files = len(filenames_)
    # feat_ = list_of_features 
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
            # st.dataframe(downloaded_indices_df)
            break
        else:
            time.sleep(10)
    end_time = time.time()
    st.write(f"Downloaded indices in {end_time - start_time} seconds")


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

    # st.markdown("recommended df:")
    # st.dataframe(recommended_df)

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
    # recommended_df
    return recommended_df

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



st.title("PSIL: Research production version")

# Input interface
st.subheader("Input Songs")
song_link = st.text_input("Enter the YouTube link of the song or playlist:")
#generate_playlist = st.checkbox("Generate spectrograms for a playlist")

if st.button("Recommend me songs"):
    with st.spinner('Processing your file(s)...'):
        if song_link:

            uploaded_df = pd.DataFrame([song_link])

            # upload this to gcs as a file called 'user_input_song.csv'

            # Create credentials object
            credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])

            # Use the credentials to create a client
            client = storage.Client(credentials=credentials)


            # The bucket on GCS in which to write the CSV file
            bucket = client.bucket('psil-app-backend-2')
            # The name assigned to the CSV file on GCS
            blob = bucket.blob('user_input_song.csv')

            # Convert the DataFrame to a CSV string with a specified encoding
            csv_string = uploaded_df.to_csv(index=False, encoding='utf-8')

            # Upload the CSV string to GCS
            blob.upload_from_string(csv_string, 'text/csv')

            st.markdown("Your song was successfully uploaded.")






            with tempfile.TemporaryDirectory() as temp_dir:

                
                
                song_names_df_path = "psil_crawler_song_names_mapped_to_latest_index.parquet.gzip"
                saved_songs_names_df = pd.read_parquet(song_names_df_path, engine="pyarrow")

                database_song_names_df = saved_songs_names_df


                old_song_names_in_order = [x for x in saved_songs_names_df.song_name.values]
                


                filtered_selection_n = 5
                

                master_links_filepath = Path("new_playlist_links_a_to_z.csv")

                links_df = pd.read_csv(master_links_filepath)
                
                top_recommendations_df = get_top_n_recommendations_gcs_version(filtered_selection_n)
                
                top_recommendations_links_df = top_recommendations_df.merge(links_df[["song_name", "song_links"]], on="song_name", how="left")[["song_name", "song_links"]].reset_index().drop(columns="index").drop_duplicates("song_name")
                # st.dataframe(top_recommendations_links_df)
                
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