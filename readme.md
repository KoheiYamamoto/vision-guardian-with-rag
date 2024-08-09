1. Start the web application for video streaming:
   - Navigate to `edge-web-camera` directory.
   - Run `streamlit run app.py`.
   - Ensure the storage connection string and container name are set correctly.

2. Start the monitoring web application:
   - Navigate to `cloud/web-logger` directory.
   - Run `python app.py`.
   - Define constants for Azure service endpoints, keys, and storage account names.
   - Set up clients for Azure services (Blob storage, Search, OpenAI).

3. Define Azure Functions to handle Blob Storage triggers:
   - Monitor the specified Blob storage container.
   - Process uploaded images and generate SAS URLs.
   - Log relevant information and perform necessary actions.

4. Ensure environment variables are used for sensitive information:
   - Avoid hardcoding credentials and keys.
   - Use environment variables for secure configuration.
   
(GH Copilot Generated)