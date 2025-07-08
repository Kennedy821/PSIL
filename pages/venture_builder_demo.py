# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:25:27 2024

@author: worldcontroller
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
from PIL import Image
from io import BytesIO
import io
import pandas as pd
import time
import polars as pl
import seaborn as sns
from PIL import Image, ImageDraw, ImageSequence, ImageFont
from google.oauth2 import service_account
from google.cloud import storage
from st_files_connection import FilesConnection
from PIL import Image
import os
from streamlit_lottie import st_lottie,st_lottie_spinner
import json
from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx
import toml
import jwt
import requests
import uuid
from datetime import datetime
from io import StringIO
import concurrent.futures

current_date = datetime.now()
formatted_date = current_date.strftime("%d_%m_%Y_%H_%M_%S")



im = Image.open('slug_logo.png')
st.set_page_config(
    page_title="PSIL",
    page_icon=im,
    initial_sidebar_state="collapsed",
    layout="wide"

    )   
# clear the cache on first load
# st.runtime.legacy_caching.clear_cache()
st.cache_resource.clear()

def stream_data(word_to_stream):
        for word in word_to_stream.split(" "):
            yield word + " "
            time.sleep(0.1)


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
                                                          &(downloaded_indices_df.comp_song.str.lower().str.contains("-"))
                                                          ]
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
    results_df = valid_results_df.sort_values("ls_distance").drop_duplicates("song_name").head(20).sort_values("ls_distance")#.head(10)
    
    # st.write(f"this search score for the top 10 valid results was: {results_df.ls_distance.mean()}")
    # st.dataframe(results_df)
    return results_df

def get_top_n_recommendations_gcs_version(n,user_hash):
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

        # st.write("this is fine:")
        # results_df
        # st.write("then something breaks:")
        # this line removes any non-audio song components                            
        results_df = results_df.merge(valid_df[["song_name"]], on="song_name", how="inner")

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
    appropriate_songs_length_check = len(recommended_df[(recommended_df["uploaded_song_components"]<2) & (recommended_df["uploaded_song_components"]>0.5)])
    if appropriate_songs_length_check>0:
        recommended_df.loc[(recommended_df["uploaded_song_components"]<2) & (recommended_df["uploaded_song_components"]>0.5),"appropriate_song"] = "true"
        recommended_df.loc[recommended_df.appropriate_song.isna(),"appropriate_song"] = "false"
    else:
        recommended_df["appropriate_song"] = "false"





    recommended_df = recommended_df[(recommended_df.appropriate_song=="true")
                                    &(recommended_df.counter<=recommended_df.total_components)].reset_index().drop(columns="index")

    # this is a check to determine if the core index has sufficient rate of clustering
    # ideally this should at least be close to 50% across all joined songs
    match_rate = recommended_df.counter.sum() / recommended_df.total_components.sum()
    # st.markdown(f"the match rate is: {round(match_rate*100,0)} %" )

    recommended_df["pct_similiar"] = recommended_df["counter"] / recommended_df["total_components"]
    recommended_df["ls_distance"] = recommended_df["uploaded_song_components"]*recommended_df["pct_similiar"]
    recommended_df = recommended_df.rename(columns={"target_song":"song_name"})
    recommended_df = recommended_df.sort_values("ls_distance", ascending=False).head(n)
    # st.dataframe(recommended_df)
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

# Access the secret key from Streamlit's secrets management
# print(st.secrets["general"])
SECRET_KEY = st.secrets["general"]["SECRET_KEY"]

# Decode and verify JWT token
def verify_token(token):
    try:
        decoded_token = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return decoded_token
    except jwt.InvalidTokenError:
        return None

# Generate JWT token after login
def generate_token(email):
    token = jwt.encode({"email": email}, SECRET_KEY, algorithm="HS256")
    return token


def get_previous_searches_fast(chosen_user):
    # Create credentials object
    credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])

    # Use the credentials to create a client
    client = storage.Client(credentials=credentials)

    # Specify your bucket name
    bucket_name = "psil-app-backend-2"

    # Get the bucket object
    bucket = client.bucket(bucket_name)

    # List all blobs in the 'users/' directory
    blobs = client.list_blobs(bucket_name, prefix=f'historic_searches/{chosen_user}')

    # Initialize an empty list to store DataFrames
    dataframes = []

    # Temporary directory for storing downloaded files (optional)
    # temp_dir = "/tmp/"  # If needed

    # Iterate over all blobs in the 'users/' directory
    for blob in blobs:
        # Check if the blob is not a directory (blob names ending with '/')
        if not blob.name.endswith('/') and ".csv" in blob.name and chosen_user in blob.name:
            # Download the blob's content as a string
            content = blob.download_as_text()

            # Read the content into a pandas DataFrame
            df = pd.read_csv(StringIO(content))
            # df["search_date"] = pd.to_datetime(df["search_date"])
            dataframes.append(df)

    output_df = pd.concat(dataframes).drop_duplicates("song_name").reset_index().sort_values("search_date", ascending=True).head(10)
    
    return output_df[["search_date","song_name",'song_link']]

# minor comment
 
def get_previous_recommendations_fast(chosen_user):
    # Create credentials object
    credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])

    # Use the credentials to create a client
    client = storage.Client(credentials=credentials)

    # Specify your bucket name
    bucket_name = "psil-app-backend-2"

    # Get the bucket object
    bucket = client.bucket(bucket_name)

    # List all blobs in the 'users/' directory
    blobs = client.list_blobs(bucket_name, prefix=f'historic_recommendations/{chosen_user}')

    # Initialize an empty list to store DataFrames
    dataframes = []

    # Temporary directory for storing downloaded files (optional)
    # temp_dir = "/tmp/"  # If needed

    # Iterate over all blobs in the 'users/' directory
    for blob in blobs:
        # Check if the blob is not a directory (blob names ending with '/')
        if not blob.name.endswith('/') and ".csv" in blob.name and chosen_user in blob.name:
            # Download the blob's content as a string
            content = blob.download_as_text()

            # Extract and parse the recommendation date from the file name
            recommendation_date_str = blob.name.split("/")[-1].split("_psil")[0]
            # st.write(recommendation_date_str)
            try:
                recommendation_date = pd.to_datetime(recommendation_date_str, format="%d_%m_%Y")
            except ValueError:
                recommendation_date = None  # Handle parsing errors if necessary

            # recommendation_date = blob.name.split("/")[-1].split("_psil")[0]

            # Read the content into a pandas DataFrame
            df = pd.read_csv(StringIO(content))
            # do a check to see if this was using fast search or deep search
            dummy_song_check = [x for x in df["anchor_song"] if "dummy" in x]
            if len(dummy_song_check)>0:
                df = df[df["anchor_song"].str.contains("dummy")]
                df["anchor_song"] = "fast_search"
            else:
                pass
            df["recommendation_date"] = recommendation_date
            # df["search_date"] = pd.to_datetime(df["search_date"])
            df = df.rename(columns={"predictions_sq":"ls_distance"})
            df = df.reset_index()[["anchor_song","comp_song","ls_distance","recommendation_date"]].sort_values(["recommendation_date","ls_distance"], ascending=[False, True]).head(10)
            dataframes.append(df)
    

    output_df = pd.concat(dataframes).sort_values(["recommendation_date","ls_distance"], ascending=[False, True]).drop_duplicates(["anchor_song","comp_song"])
    max_date_list = sorted(output_df.recommendation_date.unique(), reverse=True)[:2]
    # st.write(f"the values in the date list are : {max_date_list}")
    output_df = output_df[output_df["recommendation_date"].isin(max_date_list)]
    output_df_container = []
    for i in output_df.anchor_song.value_counts().index:
        output_int_df = output_df[output_df.anchor_song==i].sort_values(["recommendation_date","ls_distance"], ascending=[False, True]).head(10)
        output_df_container.append(output_int_df)
        del output_int_df
    output_df = pd.concat(output_df_container)

    # output_df
    
    return output_df


def check_processing_stage_1(chosen_user):
    # Create credentials object
    credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])

    # Use the credentials to create a client
    client = storage.Client(credentials=credentials)

    # Specify your bucket name
    bucket_name = "psil-app-backend-2"

    # Get the bucket object
    bucket = client.bucket(bucket_name)

    # List all blobs in the 'users/' directory for the chosen user
    blobs = client.list_blobs(bucket_name, prefix=f'users/{chosen_user}')

    # Initialize an empty list to store DataFrames
    dataframes = []

    # Iterate over all blobs in the 'users/' directory
    for blob in blobs:
        # Check if the blob is not a directory (blob names ending with '/')
        if not blob.name.endswith('/') and ".png_final" in blob.name and chosen_user in blob.name:
            # st.write(f"{blob.name} found")

            # Notify via Streamlit that checkpoint 2 is complete
            # st.write("Checkpoint 1 complete")
            pass

def check_processing_stage_2(chosen_user):
    
    # Create credentials object
    credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])

    # Use the credentials to create a client
    client = storage.Client(credentials=credentials)

    # Specify your bucket name
    bucket_name = "psil-app-backend-2"

    # Get the bucket object
    bucket = client.bucket(bucket_name)

    # List all blobs in the 'users/' directory for the chosen user
    blobs = client.list_blobs(bucket_name, prefix=f'users/{chosen_user}')

    # Initialize an empty list to store DataFrames
    dataframes = []

    # Iterate over all blobs in the 'users/' directory
    for blob in blobs:
        # Check if the blob is not a directory (blob names ending with '/')
        if not blob.name.endswith('/') and "search_list.csv" in blob.name and chosen_user in blob.name:
            # st.write(f"{blob.name} found")
            # Notify via Streamlit that checkpoint 2 is complete
            # st.write("Checkpoint 2 complete")
            pass
        
def get_url_for_song(song):
    chosen_url = search_history_df[search_history_df.song_name==song].song_link.values[0]
    # Replace this with your actual URL generation logic
    # For example, linking to Spotify search
    # return f"https://open.spotify.com/search/{song.replace(' ', '%20')}"
    return chosen_url

def get_image_for_song():

    # Assuming you have images saved locally with filenames based on the song name
    img_path = f"spotify_icon.png"
    try:
        img = Image.open(img_path)
        return img
    except FileNotFoundError:
        print(f"Image not found")
        return None


# here is the function to be used to check the checkpoints.
def wait_for_checkpoint(checkpoint_function, chosen_user, checkpoint_name, max_attempts=60, delay=5):
    """
    Waits for a specific checkpoint to complete.

    Args:
        checkpoint_function (function): The function to check the checkpoint status.
        chosen_user (str): The user's identifier.
        checkpoint_name (str): The name of the checkpoint (for logging purposes).
        max_attempts (int): Maximum number of attempts before timing out.
        delay (int): Delay (in seconds) between each check.

    Returns:
        bool: True if the checkpoint is completed within the max_attempts, False otherwise.
    """
    attempts = 0
    while attempts < max_attempts:
        try:
            checkpoint_function(chosen_user)  # Check if the checkpoint is complete
            # st.write(f"{checkpoint_name} completed.")
            return True  # If successful, return
        except Exception as e:
            # Log error (optional) and retry after delay
            st.warning(f"Waiting for {checkpoint_name} to complete. Attempt {attempts + 1}/{max_attempts}...")
            time.sleep(delay)
            attempts += 1

    st.error(f"{checkpoint_name} not completed within the timeout period.")
    return False  # Return False if checkpoint not completed





# create some cached versions of the functions that may reload repeatedly

@st.cache_data(show_spinner=False)
def get_previous_searches_fast_cached(chosen_user):
    return get_previous_searches_fast(chosen_user)

@st.cache_data(show_spinner=False)
def get_previous_recommendations_fast_cached(chosen_user, last_modified_time):
    return get_previous_recommendations_fast(chosen_user)

@st.cache_data(show_spinner=False)
def get_album_art_images():
    return [
        "https://upload.wikimedia.org/wikipedia/en/5/5f/Bon_iver.jpg",
        "https://media.pitchfork.com/photos/5935a1014fc0406ca110ccc9/master/pass/fd8402f9.jpg",
        "https://f4.bcbits.com/img/0028797410_10.jpg",
    ]

# along the same lines we're going to cache the user selected variables as the sometimes the page reloads and the user has to reselect the language and genre

# Initialize session state variables
if 'processing_type' not in st.session_state:
    st.session_state.processing_type = ""

if 'language_option' not in st.session_state:
    st.session_state.language_option = ""

if 'genre_option' not in st.session_state:
    st.session_state.genre_option = ""
# if 'uploaded_file' not in st.session_state:
#     st.session_state.uploaded_file = None
if 'search_type_option' not in st.session_state:
    st.session_state.genre_option = ""
# Step 1: Retrieve the token from the URL query parameters
# query_params = st.experimental_get_query_params()

# need to change from using st.experimental_get_query_params() to st.query_params()

# Here is the new code to get the query parameters

query_params = st.query_params.get("token")

token = st.query_params.token
if token:
    # decoded_token = verify_token(token)


# if 'token' in query_params:
#     token = query_params.get('token')[0]  # Get the token from the query
    decoded_token = verify_token(token)

    if decoded_token:
        user_email = str(decoded_token).split(":")[1].split("'")[1]
        
        st.success(f"Access granted! Welcome, {user_email}.")
        st.write(f"Your account: {user_email}")


    user_hash = token

# allow the user to type in what they are looking for 

user_input_text = st.text_input("Type in what you're looking for")
if user_input_text:


    # testing out the API itself
    response = requests.post(
        st.secrets["general"]["API_URL"],
        json={"text":user_input_text}
    )
    if response.status_code==200:
        resp_json = json.loads(response.json())

        num_results = len(resp_json["orig"])
        print(num_results)
    
        # iterate through each result to unpack the json object
        # first we'll get the recommended songs
        recommended_songs_list = []
        song_links_list = []
        artist_names_list = []
        song_names_list = []
        for idx in range(num_results):
            recommended_songs_list.append(resp_json["recommendation_songs"][str(idx)])
        # next get the links to play the songs
        for idx in range(num_results):
            song_links_list.append(resp_json["r_song_external_url"][str(idx)])
        # next get the artist names
        for idx in range(num_results):
            artist_names_list.append(resp_json["artist"][str(idx)])
        # next get the song names
        for idx in range(num_results):
            song_names_list.append(resp_json["song_name"][str(idx)])
        output_df = pd.DataFrame([artist_names_list,song_names_list,song_links_list]).T
        output_df.columns = ["artist","song","song_link"]


        # ---------------------------------------------------------------------------
        #  CSS – tweaked grid layout + nicer hover
        # ---------------------------------------------------------------------------
        st.markdown(
        """
        <style>
        .row-card{
            display:grid;
            /* avatar 56px | text grows | icon 48px  */
            grid-template-columns:56px 1fr 48px;
            align-items:center;
            gap:1rem;
            padding:.8rem 1rem;
            margin-bottom:.65rem;
            border-radius:12px;
            background:rgba(255,255,255,.06);
        }
        .avatar{
            width:56px;height:56px;border-radius:50%;
            background:#545454;color:#fff;font:700 1.25rem/56px sans-serif;
            text-align:center;overflow:hidden;
        }
        .meta{display:flex;flex-direction:column;}
        .meta .artist{margin:0;font-weight:600;font-size:1.05rem;}
        .meta .track {margin:0;opacity:.85;font-size:.95rem;}
        .wave-btn{
            width:48px;height:48px;border-radius:10px;
            display:flex;align-items:center;justify-content:center;
            transition:background .2s ease;
        }
        .wave-btn:hover{background:rgba(255,255,255,.10);}
        .wave-btn svg rect{fill:#fff;}       /* make bars white – matches dark bg   */
        </style>
        """,
        unsafe_allow_html=True
        )

        # crisp inline SVG
        wave_svg = (
            "<svg width='28' height='16' viewBox='0 0 32 18' "
            "xmlns='http://www.w3.org/2000/svg'>"
            "<rect width='3' height='18' rx='1.5'/>"
            "<rect x='6'  width='3' height='12' rx='1.5'/>"
            "<rect x='12' width='3' height='18' rx='1.5'/>"
            "<rect x='18' width='3' height='8'  rx='1.5'/>"
            "<rect x='24' width='3' height='14' rx='1.5'/>"
            "<rect x='30' width='3' height='10' rx='1.5'/>"
            "</svg>"
        )

        # ---------------------------------------------------------------------------
        #  Render each recommendation
        # ---------------------------------------------------------------------------
        for _, row in output_df.iterrows():
            # real artwork if you’ve got it, otherwise first initial
            avatar = (
                f"<img src='{row.get('artwork','')}' class='avatar'>"
                if row.get("artwork") else f"<div class='avatar'>{row.artist[0]}</div>"
            )

            # one-liner keeps Markdown from re-indenting
            card_html = (
                f"<div class='row-card'>"
                f"{avatar}"
                f"<div class='meta'><p class='artist'>{row.artist}</p>"
                f"<p class='track'>{row.song}</p></div>"
                f"<a class='wave-btn' href='{row.song_link}' target='_blank' title='Play'>{wave_svg}</a>"
                f"</div>"
            )

            st.markdown(card_html, unsafe_allow_html=True)


        # for _, row in output_df.iterrows():      # one loop → one row on screen
        #     col_artist, col_song, col_link = st.columns(3)

        #     with col_artist:
        #         st.markdown(f"### {row.artist}")

        #     with col_song:
        #         st.markdown(f"### {row.song}")

        #     with col_link:
        #         # Markdown renders the link as a button-style anchor
        #         st.markdown(f"[▶ Play]({row.song_link})")

        output_df