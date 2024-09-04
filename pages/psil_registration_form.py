import streamlit as st
import pandas as pd
import bcrypt
from google.oauth2 import service_account
from PIL import Image
from google.cloud import storage
import jwt  # To generate and decode tokens
import time
import tempfile

SECRET_KEY = st.secrets["general"]["SECRET_KEY"]

im = Image.open('slug_logo.png')
st.set_page_config(
    page_title="PSIL",
    page_icon=im,
    initial_sidebar_state="collapsed",
    ) 

def check_if_user_is_existing_user():
    if len(existing_user_list)==0:


        #download the indices from gcs
        blob = bucket.blob("error_existing_user.csv")
        if blob.exists():

            # Download the file to a destination
            blob.download_to_filename(temp_dir+"error_existing_user.csv")
            login_url = pd.read_csv(temp_dir+"error_existing_user.csv").values[0]
            existing_user_list.append(login_url)
            st.error("it looks like you've already registered with us. We are navigating you to the login page")
            time.sleep(3)
            # Redirect to external site with the token as a query parameter
            redirect_url = f"https://psilproject.streamlit.app/psil_login"
            st.markdown(f"""
            <meta http-equiv="refresh" content="0; url={redirect_url}">
            """, unsafe_allow_html=True)
            
        else:
            time.sleep(3)
            pass

# Hash the password
def hash_password(plain_text_password):
    # Generate a salt
    salt = bcrypt.gensalt()
    # Hash the password with the salt
    hashed_password = bcrypt.hashpw(plain_text_password.encode('utf-8'), salt)
    return hashed_password

# Generate JWT token after login
def generate_token(email):
    token = jwt.encode({"email": email}, SECRET_KEY, algorithm="HS256")
    return token

# Login form
st.title("Register form")

email = st.text_input("Email")
if email:
    if "@" in email:
        if ".com" in email or ".ac.uk" in email:
            pass
    else:
        st.markdown("it looks like your email isn't valid. Please try put in another email")
        

confirm_email = st.text_input("Confirm Email")
if email and confirm_email:
    if email != confirm_email:
        st.markdown("please make sure your emails match")


password = st.text_input("Password", type="password")

confirm_password = st.text_input("Confirm Password", type="password")

if password and confirm_password:
    if password != confirm_password:
        st.markdown("please make sure your passwords match!")

if st.button("Register"):
    with st.spinner("Processing your registration..."):
        with tempfile.TemporaryDirectory() as temp_dir:



            # upload registration data to user credentials database
            
            credentials_df = pd.DataFrame([email,password]).T
            credentials_df.columns = ["email","pw"]
            credentials_df["hash_pw"] = credentials_df.pw.apply(lambda x: hash_password(x).decode())
            credentials_df["token"] = credentials_df.email.apply(lambda x: generate_token(x))

            st.dataframe(credentials_df)
            credentials_df.drop(columns="pw")
            # Create credentials object
            credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])

            # Use the credentials to create a client
            client = storage.Client(credentials=credentials)


            # The bucket on GCS in which to write the CSV file
            bucket = client.bucket('psil-app-backend-2')
            # The name assigned to the CSV file on GCS
            blob = bucket.blob('new_psil_user_registration.csv')

            # Convert the DataFrame to a CSV string with a specified encoding
            csv_string = credentials_df.to_csv(index=False, encoding='utf-8')

            # Upload the CSV string to GCS
            blob.upload_from_string(csv_string, 'text/csv')

            # check to see if the email address is already registered to the site

            existing_user_list = []

            check_if_user_is_existing_user()




            time.sleep(5)
            st.success("Registration successful!")
            st.markdown("...redirecting you to the Log in page")

            # Redirect to external site with the token as a query parameter
            redirect_url = f"https://psilproject.streamlit.app/psil_login"
            st.markdown(f"""
            <meta http-equiv="refresh" content="0; url={redirect_url}">
            """, unsafe_allow_html=True)


