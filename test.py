from google.cloud import storage
from google.api_core.exceptions import Forbidden, NotFound
from google.auth import exceptions

def test_gcs_access():
    # Replace with your bucket name and blob name
    bucket_name = 'car_db'
    blob_name = 'car_db.db'

    try:
        # Initialize the client
        client = storage.Client()
        print(client)
        # Get the bucket
        bucket = client.get_bucket(bucket_name)

        print(bucket)
        # Get the blob (file) from the bucket
        blob = bucket.blob(blob_name)
    
        # Check if the blob exists
        if blob.exists():
            print(f"Blob {blob_name} found in bucket {bucket_name}")
            # Optionally download the file
            blob.download_to_filename(blob_name)
            print(f"Downloaded {blob_name} successfully.")
        else:
            print(f"Blob {blob_name} does not exist in bucket {bucket_name}")

    except Forbidden as e:
        print(f"Permission Denied: {e}")
    except NotFound as e:
        print(f"Bucket not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    test_gcs_access()
