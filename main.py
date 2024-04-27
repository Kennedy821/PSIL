import functions_framework
import numpy as np
import faiss
from google.cloud import storage
import logging
import pandas as pd
import time

# Triggered by a change in a storage bucket
@functions_framework.cloud_event
def hello_gcs(cloud_event):
    data = cloud_event.data

    event_id = cloud_event["id"]
    event_type = cloud_event["type"]

    bucket = data["bucket"]
    name = data["name"]
    metageneration = data["metageneration"]
    timeCreated = data["timeCreated"]
    updated = data["updated"]

    print(f"Event ID: {event_id}")
    print(f"Event type: {event_type}")
    print(f"Bucket: {bucket}")
    print(f"File: {name}")
    print(f"Metageneration: {metageneration}")
    print(f"Created: {timeCreated}")
    print(f"Updated: {updated}")
    
    main_function_psil(data, context)




def search_faiss(data, index_path, bucket_name):
    # Load your FAISS index (this is a placeholder, adjust based on your index type)
        # Create a client to access the GCS bucket
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(index_path)
    
    # Download the file to a local temp path
    local_path = f"/tmp/{index_path.split('/')[-1]}"
    blob.download_to_filename(local_path)
    index = faiss.read_index(local_path)
    logging.warning("index loaded successfully")
    feat_ = data
    logging.warning(f"Numpy array feature shape is: {feat_.shape}")
    list_of_indices = []
    for i in range(len(feat_)):
        query_vector = feat_[i]
        logging.warning(f"QUERY VECTOR SHAPE IS:{query_vector.shape}")
        query_vector = query_vector.reshape(1,query_vector.shape[0])
        # Perform the search
        D, I = index.search(query_vector, k=10)  # Example: search for the 10 nearest neighbors
        list_of_indices.append(pd.DataFrame(I))
    output_df = pd.concat(list_of_indices)
    # The name assigned to the CSV file on GCS
    blob = bucket.blob('queried_indices.csv')
    blob.upload_from_string(output_df.to_csv(index=False), 'text/csv')
    return D, I

def read_numpy_data(bucket_name, file_path):
    # Create a client to access the GCS bucket
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_path)
    
    # Download the file to a local temp path
    local_path = f"/tmp/{file_path.split('/')[-1]}"
    blob.download_to_filename(local_path)
    
    # Load the numpy .dat file
    data = np.load(local_path,allow_pickle=True)
    return data

def read_memmap(bucket_name, file_path):
    # Create a client to access the GCS bucket
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_path)
    
    # Download the file to a local temp path
    local_path = f"/tmp/{file_path.split('/')[-1]}"
    blob.download_to_filename(local_path)
    
    assumed_shape = 327680
    assumed_number_of_rows = int(len(np.memmap(local_path, dtype='float32', mode='r'))/assumed_shape)
    logging.warning(f"THE ASSUMED SHAPE OF THIS ARRAY IS:{assumed_number_of_rows}")
    # Load the np.memmap file
    mmap = np.memmap(local_path, dtype='float32', mode='r', shape=(assumed_number_of_rows,assumed_shape))
    return mmap

def write_results_to_gcs(bucket_name, file_name, data):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    
    # Convert your data to bytes and upload
    blob.upload_from_string(data.tobytes())
def delete_blob(bucket_name, blob_name):
    """Deletes a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.delete()

    print(f"Blob {blob_name} deleted.")


def main_function_psil(data, context):
    # You can pass the bucket and file names as part of the request or hard code them
    bucket_name = data['bucket']
    file_name = data['name']

    numpy_file_1 = "uploaded_song_feature_vector.dat"
    
    # Read numpy files
    data_1 = read_memmap(bucket_name, numpy_file_1)
    logging.warning("file: feature vector numpy array successfully loaded")
    index_path = "faiss_quantised_testing_1000_clusters.index"
    # Run FAISS search
    distances, indices = search_faiss(data_1,index_path,bucket_name)  # Assuming data_2 is an index
    
    time.sleep(5)
    # Example usage
    #delete_blob(bucket_name, f'{bucket_name}/queried_indices.csv')

    # Write results back to GCS
    result_file_name = "search_results"
    write_results_to_gcs(bucket_name, result_file_name, indices)
    logging.warning("files have been written to gcs")
    return f"Results stored in {result_file_name}"
