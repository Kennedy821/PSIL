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
import os
from streamlit_lottie import st_lottie,st_lottie_spinner
import json
# from torchvision import transforms
# import re
# from fuzzywuzzy import fuzz
# from fuzzywuzzy import process
from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx
import toml
from datetime import datetime
import jwt

current_date = datetime.now()
formatted_date = current_date.strftime("%d_%m_%Y")


im = Image.open('slug_logo.png')
st.set_page_config(
    page_title="PSIL",
    page_icon=im,
    initial_sidebar_state="collapsed",
    )     

# Custom CSS to ensure dark theme and hide footer
# st.markdown("""
#     <style>
#         .stApp {
#             background-color: black;
#         }
#     </style>
# """, unsafe_allow_html=True)

# st.markdown("""
# <style>
#     .stApp {
#         background-color: black;
#     }
#     .stApp > header {
#         background-color: transparent;
#     }
#     .stMarkdown, .stText, .stCode, .stTextInput > div > div > input {
#         color: white !important;
#     }
# </style>
# """, unsafe_allow_html=True)
SECRET_KEY = st.secrets["general"]["SECRET_KEY"]


def get_top_n_recommendations_gcs_version_new(n,user_hash):


    while True:

        # check_processing_stage_1(user_hash)

        # check_processing_stage_2(user_hash)

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
        blob = bucket.blob(f"users/{user_hash}/combined_similarity_results.csv")
        if blob.exists():

            # Download the file to a destination
            blob.download_to_filename(temp_dir+"combined_similarity_results.csv")
            downloaded_indices_df = pd.read_csv(temp_dir+"combined_similarity_results.csv")
            downloaded_indices_df["target_song"] = downloaded_indices_df["comp_song"]
            downloaded_indices_df = downloaded_indices_df[~(downloaded_indices_df.comp_song.str.lower().str.contains("review"))
                                                          &~(downloaded_indices_df.comp_song.str.lower().str.contains("tribute"))
                                                          &~(downloaded_indices_df.comp_song.str.lower().str.contains("react"))
                                                          &(downloaded_indices_df.comp_song.str.lower().str.contains("-"))]
            recommended_df = downloaded_indices_df.copy()
            # st.dataframe(downloaded_indices_df)
            break
        else:
            time.sleep(10)
    end_time = time.time()
    # st.write(f"Downloaded indices in {end_time - start_time} seconds")

    # st.write(f"this search score was: {downloaded_indices_df.predictions_sq.mean()}")
    # st.write(f"this search score for the top 10 was: {downloaded_indices_df.sort_values('predictions_sq', ascending=True).head(10).predictions_sq.mean()}")

    # this is the filtering according to the user's language and genre preferences

    total_components_df = database_song_names_df.copy()
    total_components_df["target_song"] = total_components_df["song_name"].str.split("_spect").str[0]
    # total_components_df
    total_components_df["total_components"] = 1
    total_components_df = total_components_df[["target_song","total_components",language_option.lower()]].groupby(["target_song"]).sum().sort_values("total_components", ascending=False).reset_index()
    
    language_df = total_components_df.copy().drop(columns="total_components")
    language_df.loc[language_df[language_option.lower()]>0,language_option.lower()] = 1

    # st.markdown("this is the language df")
    # st.dataframe(language_df)

    recommended_df = recommended_df.merge(language_df, on="target_song")


    if language_option=="All":
        pass
    else:
        recommended_df = recommended_df[recommended_df[language_option.lower()]==1]


    # st.markdown("this is the updated recommended df with a language flag")
    # st.dataframe(recommended_df)

    # st.markdown("this is the genre df")
    # st.dataframe(genre_df)

    # this is being disabled as genre filtering has been moved to the backend 
    # ---------------------------------------------------------------------------


    # recommended_df = recommended_df.merge(genre_df, on="target_song")


    # if genre_option=="All":
    #     pass
    # else:
    #     recommended_df = recommended_df[recommended_df["target_song"].isin(in_scope_genre_song_names)]

    # ---------------------------------------------------------------------------

    # this is the valid df
    # st.write("this is the valid df")
    # st.dataframe(valid_df)
    valid_df["song_name"] = valid_df["song_name"].str.split("_spect").str[0]

    recommended_df = recommended_df.rename(columns={"comp_song":"song_name","predictions_sq":"ls_distance"}).drop(columns="anchor_song")
    # do this merge if you want to filter the results against a pre-made list of valid songs
    valid_results_df = recommended_df#.merge(valid_df[["song_name"]], on="song_name", how="inner")
    # st.write("this is the recommended df  after merging with the valid results df")
    # st.dataframe(valid_results_df)
    results_df = valid_results_df.sort_values("ls_distance").drop_duplicates("song_name").head(20).sort_values("ls_distance").head(10)
    
    # st.write(f"this search score for the top 10 valid results was: {results_df.ls_distance.mean()}")
    # st.dataframe(results_df)
    return results_df

# Generate JWT token after login
def generate_token(email):
    token = jwt.encode({"email": email}, SECRET_KEY, algorithm="HS256")
    return token

def stream_data(word_to_stream):
    for word in word_to_stream.split(" "):
        yield word + " "
        time.sleep(0.25)


def get_top_n_recommendations_gcs_version(n, user_hash):
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
        blob = bucket.blob(f"users/{user_hash}/queried_indices.csv")
        if blob.exists():

            # Download the file to a destination
            blob.download_to_filename(temp_dir+"queried_indices.csv")
            downloaded_indices_df = pd.read_csv(temp_dir+"queried_indices.csv")
            # st.dataframe(downloaded_indices_df)
            break
        else:
            time.sleep(10)
    end_time = time.time()
    # st.write(f"Downloaded indices in {end_time - start_time} seconds")


    total_components_df = database_song_names_df.copy()
    total_components_df["target_song"] = total_components_df["song_name"].str.split("_spect").str[0]
    # total_components_df
    total_components_df["total_components"] = 1
    total_components_df = total_components_df[["target_song","total_components",language_option.lower()]].groupby(["target_song"]).sum().sort_values("total_components", ascending=False).reset_index()
    

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
        pivoted_df = pivoted_df.merge(total_components_df, on="target_song").drop(columns=f'{language_option.lower()}')
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
    # polars version
    # recommended_df = pl.from_pandas(recommended_df).groupby(["origin_song","target_song"]).agg(pl.col("counter").sum(),
    #                                                                                            pl.col("total_components").median()).to_pandas().sort_values("counter", ascending=False).reset_index()
    
    # pandas version 
    recommended_df = (recommended_df.groupby(["origin_song", "target_song"])
                  .agg(counter=("counter", "sum"),
                       total_components=("total_components", "median"))
                  .sort_values("counter", ascending=False)
                  .reset_index())
    
    language_df = total_components_df.copy().drop(columns="total_components")
    language_df.loc[language_df[language_option.lower()]>0,language_option.lower()] = 1

    # st.markdown("this is the language df")
    # st.dataframe(language_df)
    recommended_df = recommended_df.merge(language_df, on="target_song")

    # st.markdown("this is the updated recommended df with a language flag")
    # st.dataframe(recommended_df)

    if language_option=="All":
        pass
    else:
        recommended_df = recommended_df[recommended_df[language_option.lower()]==1]


    # st.markdown("this is the updated recommended df with a language flag")
    # st.dataframe(recommended_df)

    # st.markdown("this is the genre df")
    # st.dataframe(genre_df)

    recommended_df = recommended_df.merge(genre_df, on="target_song")


    if genre_option=="All":
        pass
    else:
        recommended_df = recommended_df[recommended_df["target_song"].isin(in_scope_genre_song_names)]

    # st.markdown("this is the updated recommended df with a genre flag")
    # st.dataframe(recommended_df)

    recommended_df["origin_song_counter"] = total_uploaded_files
    recommended_df["uploaded_song_components"] = recommended_df["total_components"] / recommended_df["origin_song_counter"]
    recommended_df.loc[(recommended_df["uploaded_song_components"]<2) & (recommended_df["uploaded_song_components"]>0.5),"appropriate_song"] = "true"
    recommended_df.loc[recommended_df.appropriate_song.isna(),"appropriate_song"] = "false"



 

    recommended_df = recommended_df[(recommended_df.appropriate_song=="true")
                                    &(recommended_df.counter<=recommended_df.total_components)].reset_index().drop(columns="index")

    # this is a check to determine if the core index has sufficient rate of clustering
    # ideally this should at least be close to 50% across all joined songs
    match_rate = recommended_df.counter.sum() / recommended_df.total_components.sum()
    # st.markdown(f"the match rate is: {round(match_rate*100,0)} %" )

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



# st.title("PSIL: Research production version")

with open("purple_pink_ball_gradient.json", "r") as f:


    lottie_json = json.load(f)

# Input interface
st.subheader("Input Songs")
song_link = st.text_input("Enter the SoundCloud link of the song you'd like to get recommendations for:")

st.write("Select which language you'd like your song results to be in")

language_option = st.selectbox(
    'Select Language for your recommendations',
    ('', 'All', 'English','French','Japanese')  # Add an empty string as the first option
)

# Display the selected option
if language_option:
    st.write('You selected:', language_option)
else:
    st.write('Please select a language.')



genre_option = st.selectbox(
    'Select Genre for your recommendations',
    ('','All', 'Rock', 'Hip-Hop','Electronic','Folk','Experimental',"Instrumental","Pop")  # Add an empty string as the first option
)

# Display the selected option
if genre_option:
    st.write('You selected:', genre_option)
else:
    st.write('Please select a genre.')

primaryColor = toml.load(".streamlit/config.toml")['theme']['textColor']
s = f"""
<style>
div.stButton > button:first-child {{ border: 5px solid {primaryColor}; border-radius:20px 20px 20px 20px; }}
<style>
"""
st.markdown(s, unsafe_allow_html=True)
if st.button("Recommend me songs"):
    with st.spinner('Processing your recommendations...this usually takes around 5 minutes.'):
        

    # st.write("Processing your link...")



    # with st_lottie_spinner(lottie_json, speed=3, height=750,width=750):


    # # Initialize session state for controlling the animation
    # if "show_animation" not in st.session_state:
    #     st.session_state.show_animation = True

    # # Display the animation if the state is set to True
    # if st.session_state.show_animation:


            # animation_object = st_lottie(lottie_json,speed=2, height=750, width=750)

            

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

            # st.markdown("Your song was successfully uploaded.")
            import uuid
            unique_id = uuid.uuid4()
            clean_token = generate_token("demo")

            # this makes sure that requests are segregated by each user
            user_directory = f'users/{clean_token}/'

            logging_filename = f"{formatted_date}_psil_site_search_{clean_token}_{unique_id}_fast_search.csv"
            full_file_path = f'{user_directory}{logging_filename}'

            # logging_df = pd.DataFrame([str(decoded_token),song_link]).T

            logging_df = pd.DataFrame([str("demo")])
            logging_df.columns = ["user"]
            logging_df["song_link"] = str(song_link)


            # logging_df = pd.DataFrame([{'user': decoded_token, 'song_link': song_link}])

            # logging_df.columns = ["user","song_link"]
            # st.dataframe(logging_df)


            
            # The bucket on GCS in which to write the CSV file
            bucket = client.bucket('psil-app-backend-2')
            # The name assigned to the CSV file on GCS
            blob = bucket.blob(full_file_path)

            # Convert the DataFrame to a CSV string with a specified encoding
            csv_string = logging_df.to_csv(index=False, encoding='utf-8')

            # Upload the CSV string to GCS
            blob.upload_from_string(csv_string, 'text/csv')






            with tempfile.TemporaryDirectory() as temp_dir:

                valid_df = pd.read_parquet("psil_crawler_song_names_mapped_as_valid_songs_or_not.parquet.gzip")
                valid_df = valid_df[valid_df["valid_song"]=='1']

                
                
                genre_df = pd.read_parquet("majority_genre_for_song.parquet.gzip").rename(columns={"song_name":"target_song"})
                genre_df["target_song"] = genre_df["target_song"].str.split("_spect").str[0]
                # st.markdown(f"the columns in genre_df are: {genre_df.columns}")
                genre_df = genre_df[genre_df[genre_option]>0]
                in_scope_genre_song_names = [x for x in genre_df.target_song.values]


                
                df_container = []
                for file_ in os.listdir(os.getcwd()):
                    if "w_languages_" in file_:
                        int_df = pd.read_parquet(file_, engine="pyarrow")
                        int_df.iloc[:,1:] = int_df.iloc[:,1:].astype(float)
                        int_df = int_df.set_index("song_name")
                        # st.dataframe(int_df.head())

                        df_container.append(int_df)
                        del int_df

                database_song_names_df = pd.concat(df_container, axis=1).reset_index()[["song_name",language_option.lower()]]
                # st.dataframe(database_song_names_df)
                del df_container
                # database_song_names_df

                # song_names_df_path = "psil_crawler_song_names_mapped_to_latest_index_w_languages.parquet.gzip"
                # saved_songs_names_df = pd.read_parquet(song_names_df_path, engine="pyarrow")

                # saved_songs_names_df.iloc[:,1:] = saved_songs_names_df.iloc[:,1:].astype(float)
                # saved_songs_names_df


                # database_song_names_df = saved_songs_names_df


                old_song_names_in_order = [x for x in database_song_names_df.song_name.values]
                


                filtered_selection_n = 5
                

                master_links_filepath = Path("new_playlist_links_a_to_z.csv")

                links_df = pd.read_csv(master_links_filepath)
                
                # top_recommendations_df = get_top_n_recommendations_gcs_version(filtered_selection_n, clean_token)
                top_recommendations_df = get_top_n_recommendations_gcs_version_new(filtered_selection_n, clean_token)

                
                top_recommendations_links_df = top_recommendations_df.merge(links_df[["song_name", "song_links"]], on="song_name", how="left")[["song_name", "song_links"]].reset_index().drop(columns="index").drop_duplicates("song_name")

                

                
                # st.dataframe(top_recommendations_links_df)
                
                # Create a dictionary mapping song names to links
                song_links_map = dict(zip(top_recommendations_links_df['song_name'], top_recommendations_links_df['song_links']))
                
                markdown_list_items = []
                markdown_list_items_no_links = []
                for song in top_recommendations_df['song_name']:
                    song_len = len(song)
                    if song_len > 10000:
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
                




                st.session_state.show_animation = False 


                # Display in Streamlit
                st.header("Here are your recommendations")
                
                #gif_with_text = display_animated_text(gif_path,markdown_list_no_links)
                
                st.write_stream(stream_data(markdown_list))

                
                with st.expander("See how much we think you'll like these based on your uploaded song"): 
                    starting_value = 0  # Your starting/reference value
                    values = top_recommendations_df.sort_values("ls_distance", ascending=False).ls_distance  # Individual values to compare
                    labels = [x for x in top_recommendations_df.sort_values("ls_distance", ascending=False).song_name.values]
                    #song_names_markdown_list = ""
                    #st.markdown(labels)
                
                    fig, ax = plt.subplots(figsize=(5, 10))
                
                    fig.patch.set_facecolor('#2D3250')
                    ax.set_facecolor('#2D3250')
                    # Plotting each point with a line to the starting value
                    for i, value in enumerate(values):
                        ax.plot([starting_value, value],[labels[i], labels[i]], 'grey')  # Line
                        ax.plot(value, labels[i] , 'o', color='#F5E8C7')  # Dot
                    
                    # Highlight the starting value across the chart
                    #ax.axvline(starting_value, color='red', linestyle='--', label='Chosen song')
                    
                    #plt.title('Proximity to Starting Value')
                    # plt.xticks(fontsize=15, rotation=45)

                    # Set x and y axis text color
                    ax.tick_params(axis='x', colors='#F5E8C7')  # Red color for x-axis text
                    ax.tick_params(axis='y', colors='#F5E8C7')  # Green color for y-axis text

                    # Set axis labels
                    ax.set_xlabel('X Axis', color='#2D3250')  # Red label for x-axis
                    ax.set_ylabel('Y Axis', color='#2D3250')  # Green label for y-axis

                    # Remove x-ticks
                    plt.xticks([])

                    # Remove x-labels
                    plt.gca().xaxis.set_ticklabels([])

                    # Set the y-tick labels font to sans-serif
                    for label in ax.get_yticklabels():
                        label.set_fontname('sans-serif')

                    plt.yticks(fontsize=20)
                    plt.legend()
                    sns.despine()
                    st.pyplot(fig)
        
                
            
            
            
            
        else:
            st.warning("Please enter the YouTube link of the song you'd like to get recommendations for.")