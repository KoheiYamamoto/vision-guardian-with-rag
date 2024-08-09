from streamlit_webrtc import webrtc_streamer
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import streamlit as st, av, cv2, time 

### CREDENTIALS ###
STORAGE_CONNECTION_STRING = ""
STORAGE_CONTAINER_NAME = "imgs"

st.title("Edge Camera")
st.write("Show what you want to monitor")

# Create a blob client
blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)

last_saved_time = time.time()
def callback(frame):
    global last_saved_time

    img = frame.to_ndarray(format="bgr24")

    current_time = time.time()
    if current_time - last_saved_time >= 15:  # 10 seconds have passed        
        # Convert the image to bytes
        is_success, im_buf_arr = cv2.imencode(".jpg", img)
        byte_im = im_buf_arr.tobytes()
        blob_client = blob_service_client.get_blob_client(STORAGE_CONTAINER_NAME, f"frame_{int(current_time)}.jpg")
        # Upload the image to Azure Blob Storage
        blob_client.upload_blob(byte_im)

        last_saved_time = current_time  # update the last saved time
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="example", video_frame_callback=callback)