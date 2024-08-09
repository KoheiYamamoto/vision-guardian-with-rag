from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.cosmos import CosmosClient
from azure.storage.blob import BlobServiceClient, ContainerSasPermissions, generate_container_sas
from azure.core.credentials import AzureKeyCredential
from datetime import datetime, timedelta
from azure.functions import InputStream
from PIL import Image, ImageDraw, ImageFont
import azure.functions as func, logging, requests, json, uuid, cv2, matplotlib.pyplot as plt, io, numpy as np

### CREDENTIALS ###
AOAI_LLM_ENDPOINT = "" 
AOAI_LLM_TURBO_KEY = ""
AOAI_LLM_API_VERSION = ""
AOAI_LLM_DEPLOYMENT_NAME = ''

AOAI_LMM_ENDPOINT = "" 
AOAI_LMM_TURBO_KEY = ""
AOAI_LMM_API_VERSION = ""
AOAI_LMM_DEPLOYMENT_NAME = ''

CV_ENDPOINT = ""
CV_KEY = ""
SEARCH_ENDPIONT = ""
SEARCH_KEY = ""
SEARCH_INDEX = ""
KB_FIELDS_CONTENT = ""
KB_FIELDS_SOURCEPAGE = ""
STORAGE_ACCOUNT_NAME = ""
STORAGE_CONTAINER_NAME = ""
STORAGE_CONTAINER_CAPTIONED_NAME = ""
STORAGE_KEY = ""
STORAGE_CONNECTION_STRING = ""
COSMOS_ENDPOINT = ""
COSMOS_KEY = ""
COSMOS_DATABASE_NAME = ""
COSMOS_CONTAINER_NAME = ""

# set up clients
credential = AzureKeyCredential(SEARCH_KEY)
blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
search_client = SearchClient(endpoint=SEARCH_ENDPIONT,
                             index_name=SEARCH_INDEX,
                             credential=credential)
client = AzureOpenAI(
            azure_endpoint = AOAI_LLM_ENDPOINT, 
            api_key = AOAI_LLM_TURBO_KEY,  
            api_version = AOAI_LLM_API_VERSION, 
        )
client_vision = AzureOpenAI(
            api_key=AOAI_LMM_TURBO_KEY,  
            api_version=AOAI_LMM_API_VERSION,
            base_url=f"{AOAI_LMM_ENDPOINT}openai/deployments/{AOAI_LMM_DEPLOYMENT_NAME}/extensions",
        )

def get_sas_url(CONTAINER_NAME, blob_name):
    blob_url = "https://" + STORAGE_ACCOUNT_NAME + ".blob.core.windows.net/" +  blob_name
    logging.info("Getting a blob_url for the image...: %s", blob_url)
    sas = generate_container_sas(account_name=STORAGE_ACCOUNT_NAME, account_key=STORAGE_KEY, container_name=CONTAINER_NAME, 
            permission=ContainerSasPermissions(add=True, create=True, write=True, read=True),
            expiry=datetime.utcnow() + timedelta(days=60))
    sas_url = blob_url + "?" + sas
    return sas_url

def get_description(image_url):
    prompt_system = """
    You are a prompt surveillance system. Response speed is most important. Please follow the follwoing instructions:
    - You tasks is to detect potential dangers e.g. "a person has a knife in the factory and there is a girl behind", "there is a crowd around a factory machine."
        - You only need to detect the potential dangers. You don't need to provide any advice or reasons.
    - If there are multiple objects, describe the most dangerous one.
    - If there are no dangerous objects, describe the scene. e.g. "a person is sitting on a chair."
        - You don't have to mention if there any dangers. Just describe the scene.
    """

    response = client_vision.chat.completions.create(
        model=AOAI_LMM_DEPLOYMENT_NAME,
        messages=[
            { "role": "system", "content": prompt_system },
            { "role": "user", "content": [  
                { 
                    "type": "text", 
                    "text": "Detect dangers in the image." 
                },
                { 
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
            ] } 
        ],
        max_tokens=2000 
    )
    response = response.choices[0].message.content 
    return response

def get_caption(image_url):
    headers={
        'Content-Type': 'application/json',
        'Ocp-Apim-Subscription-Key': CV_KEY
    }
    response =  requests.post(CV_ENDPOINT + "computervision/imageanalysis:analyze?api-version=2023-02-01-preview&features=denseCaptions&features&language=en&gender-neutral-caption=False", params={}, headers=headers, json={"url": image_url})
    caption_depth_results = json.loads(response.content)["denseCaptionsResult"]["values"]
    return caption_depth_results

def overlay_caption(image_url, caption_depth_en):
    response = requests.get(image_url)
    image_file = io.BytesIO(response.content)
    image = Image.open(image_file) # Load the image with PIL
    image = np.array(image) # Convert the image to a NumPy array
    # For each entry in densecaption_result, draw a rectangle and put a caption
    for entry in caption_depth_en:
        bbox = entry['boundingBox']
        text = entry['text']
        confidence = entry['confidence']

        # The bounding box coordinates are given as x, y, width, height
        # We need to convert them to the top-left and bottom-right coordinates
        top_left = (bbox['x'], bbox['y'])
        bottom_right = (bbox['x'] + bbox['w'], bbox['y'] + bbox['h'])
        # Draw the rectangle
        image = cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        # Calculate the size of the text
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        # Get the top-left corner of the text background
        bg_tl = (bbox['x'], bbox['y'] + bbox['h'] - text_size[1])
        # Get the bottom-right corner of the text background
        bg_br = (bbox['x'] + text_size[0], bbox['y'] + bbox['h'])
        # Convert the OpenCV image (in BGR format) to PIL image (in RGB format)
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Create an ImageDraw object
        draw = ImageDraw.Draw(image_pil, "RGBA")
        # Draw a semi-transparent rectangle as the background of the text
        draw.rectangle([bg_tl, bg_br], fill=(0, 255, 0, 128))
        # Convert the PIL image back to OpenCV image
        image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        # Put the caption in gray at the bottom-left of the rectangle with smaller font size
        image = cv2.putText(image, text, (bbox['x'], bbox['y'] + bbox['h']), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return image

def upload_captioned_image(blob_captioned_name, image):
    # Convert the image back to BGR format for saving
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Encode the image into jpeg format
    is_success, buffer = cv2.imencode(".jpg", image_bgr)
    # Convert to bytes
    image_bytes = buffer.tobytes()
    # Create a blob service client
    blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
    # Create a blob client
    blob_client = blob_service_client.get_blob_client(STORAGE_CONTAINER_CAPTIONED_NAME, blob_captioned_name)
    # Upload the image to the blob
    blob_client.upload_blob(image_bytes, overwrite=True)

def fucntion_calling(caption_en):
    # Step 1: send the conversation and available functions to the model
    prompt_system = """
    Your task is to find the risk hedge measurement only if needed to avoid potential dangers in the situation.
    Please follow the below instructions:
    - If the level of danger is more than 7, then search for the similar situation in the database to avoid potential dangers in the situation. The below is the example of the database response.
    {"measurement": "take the lighter away", "level": 8, "situation": "a person holding lighter", "reason": "the person may light a fire"}
    - If the level of danger is less than 8, then no need to search for the similar situation in the database.
    {"measurement": "n/a", "level": 1, "situation": "dog and cat are playing", "reason": "n/a"}
    Note that the level of danger is from 0 to 10. 
    Also, you need to provide the reason of the danger, e.g. the person may attack somebody else with the knife.
    """
    messages = [{"role": "system", "content": prompt_system},
                {"role": "user", "content": "Are there any dander in the situation?: " + "a person is running"},
                {"role": "assistant", "content": "{'measurement': 'Alert security to check the person', 'level': '8', 'situation': 'a person is running', 'reason': 'the person may be slipping and falling'}"},
                {"role": "user", "content": "Are there any dander in the situation?: " + "a person holding a knife"},
                {"role": "assistant", "content": "{'measurement': 'Take the knife away', 'level': '9', 'situation': 'a person holding a knife, 'reason': 'the person may attack somebody else with the knife'}"},
                {"role": "user", "content": "Are there any dander in the situation?: " + "a person taking a selfie"},
                {"role": "assistant", "content": "{'measurement': 'Alert the person for the security', 'level': '8', 'situation': 'a person taking a selfie', 'reason': 'the person is unaware of the surrounding'}"},
                {"role": "user", "content": "Are there any dander in the situation?: " + "a person drinking water"},
                {"role": "assistant", "content": "{'measurement': 'Alert the person for the security', 'level': 'n/a', 'situation': 'a person drinking water', 'reason': 'n/a'}"},
                {"role": "user", "content": "Are there any dander in the situation?: " + caption_en}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "rag",
                "situation": "Search for the similar situation in the database to avoid potential dangers in the situation, when level of danger is more than 7.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "situation": {
                            "type": "string",
                            "description": "The situation of the danger, e.g. a person holding a knife",
                        },
                        "level": {
                            "type": "integer",
                            "description": "The level of danger, from 0 to 10 e.g. 9 for a person holding a knife",
                        },
                        "reason": {
                            "type": "string",
                            "description": "The reason of the danger, e.g. the person may attack somebody else with the knife",
                        }
                    },
                    "required": ["situation", "level", "reason"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "do_nothing",
                "situation": "Do nothing since any potential dangaers are detected in the situation, when level of danger is less than 8.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "situation": {
                            "type": "string",
                            "description": "The situation, if any e.g. a cat and dog are playing",
                        },
                        "level": {
                            "type": "integer",
                            "description": "The level of danger, from 0 to 10 e.g. 1 for the cat and dog playing",
                        },
                        "reason": {
                            "type": "string",
                            "description": "The reason of the indanger, i.e.. n/a",
                        }
                    },
                    "required": ["situation", "level", "reason"],
                },
            },
        }
    ]

    response = client.chat.completions.create(
        model=AOAI_LLM_DEPLOYMENT_NAME,
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "rag": rag,
            "do_nothing": do_nothing,
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's reply

        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                situation=function_args.get("situation"),
                level=function_args.get("level"),
                reason=function_args.get("reason")
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response 
        
        # result of RAG function
        # print(function_response)

        # second_response = client.chat.completions.create(
        #     model=AOAI_LLM_DEPLOYMENT_NAME,
        #     messages=messages,
        # )  # get a new response from the model where it can see the function response
        # return second_response.choices[0].message

        print("--- Conlusion ---")
        if json.loads(function_response)["measurement"] == "n/a":
            print("AI has concluded that there is no danger.")
            return function_response
        elif json.loads(function_response)["measurement"] == "undefined":
            print("AI was unable to identify a match in the database but confirmed the presence of danger.")
            return function_response
        else:
            print("AI has confirmed the presence of a danger matching the database.")
            return function_response

def do_nothing(situation, level, reason):
    """Do nothing since any potential dangaers are detected in the situation, when level of danger is less than 5."""
    print("Situation:", situation)
    print("Level:", level)
    print(">>> Action: AI has determined that no specific action is required.")
    print(">>> Reason:", reason)
    print("--- Function Calling: do_nothing ---")
    return json.dumps({"measurement": "n/a", "level": level, "situation": situation, "reason": "n/a"})

def rag(situation, level, reason):
    """Search for the similar situation in the database to avoid potential dangers in the situation, when level of danger is more than 4."""
    print("Situation:", situation)
    print("Level:", level)
    print(">>> Action: AI has determined that a RAG (Retrieval from Database) search is required.")
    print(">>> Reason:", reason)
    print("--- Function Calling: rag ---")

    search_results = search_client.search(situation, top=3)
    search_results = [result[KB_FIELDS_CONTENT] for result in search_results]
    content = "\n".join(search_results)

    prompt_system = """
        You are an AI assisntant and advice the user to avoid dangers based on the relevant search results. You must follows the below examples:
        - search results (situation -> measurement): 
        "a person has a knife -> take the knife away"
        "a person is lying on the ground -> call somebody"
        "a person with long hair > Alert security to check the person"
        - user message: "situation: "a person holding lighter"
        - your final output: 
        {
        "measurement": "take the lighter away"
        }
        When you cannot find any relevant information, the output should be otherwise as follows:
        {
        "measurement": "undefined"
        }
        """
    prompt_user = """
        - search results (situation -> measurement): 
        {content}
        - situation: {user_input}
        """
    messages = [{"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user.format(content=content, user_input=situation)}]
    # print(messages)
    response = client.chat.completions.create(
        model=AOAI_LLM_DEPLOYMENT_NAME,
        messages=messages)
    # print(response.choices[0].message)
    response = json.loads(response.choices[0].message.content)["measurement"]
    print("RAG Result:", response)
    if response == "undefined":
        return json.dumps({"measurement": "undefined", "level": level, "situation": situation, "reason": reason})
    else:
        return json.dumps({"measurement": response, "level": level, "situation": situation, "reason": reason})

# for debugging
# class myblob:
#     name = "imgs/frame_1707454387.jpg"
#     length = 1000
# myblob = myblob()

def main(myblob: InputStream):
# def main():
    logging.info(f"Python blob trigger function processed blob \n"
                 f"Name: {myblob.name}\n"
                 f"Blob Size: {myblob.length} bytes")
    
    # generate sas url for the image
    sas_url = get_sas_url(STORAGE_CONTAINER_NAME, myblob.name)
    logging.info("Getting a sas_url for the image...:%s", sas_url)    
    
    ### swap this line with caption api if really needed
    print("Gettting a description for the image...")
    description = get_description(sas_url)
    logging.info("Getting a description for the image...:%s", description)

    ### swap this line with turbo with vision api if really needed
    caption_depth_en = get_caption(sas_url)
    caption_en = caption_depth_en[0]["text"]
    logging.info("Getting a caption for the image...:%s", caption_en)

    # overlay the captions on the image
    image_captioned = overlay_caption(sas_url, caption_depth_en)
    # save it to a new blob in the same container and get the url 
    image_captioned_name = myblob.name.split("/")[-1].split(".")[0] + "_captioned.jpg"
    upload_captioned_image(image_captioned_name, image_captioned)
    sas_url_captioned = get_sas_url(STORAGE_CONTAINER_CAPTIONED_NAME, STORAGE_CONTAINER_CAPTIONED_NAME + "/" + image_captioned_name)

    try:
        # get the response from function calling including RAG
        response = fucntion_calling(description)
        response = json.loads(response)
    except Exception as e: # error varies: content safety, too many requests, etc.
        print("Fucntion callling error: ", e)
        response = {"measurement": "error", "situation": "error", "level": "error", "reason": "error"}

    # upsert the response to cosmos db
    client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)    
    database = client.get_database_client(COSMOS_DATABASE_NAME)
    container = database.get_container_client(COSMOS_CONTAINER_NAME)
    item = {
        "id": str(uuid.uuid4()),
        "image_url": sas_url,
        "image_captioned_url": sas_url_captioned,
        "description": description,
        "situation": response["situation"],
        "situation_depth": caption_depth_en,
        "measurement": response["measurement"],
        "level": response["level"],
        "reason": response["reason"],
        "time": myblob.name.split("/")[-1].split("_")[1].split(".")[0]
    }
    container.upsert_item(item)
    # print(item)

# main()

# RAG テスト用
# Person shows dances > Alert security to check the person
# Poorly lit areas > improve lighting installation
# Overcrowding in specific areas > manage worker certain areas 
# person pulls someone's arm > Alert security to check the person
# a person is running > Alert security to check the person
# Workers not wearing safety gear or helmet > Reminder sent to employee's terminal to wear safety gear
# a person with long hair > Alert security to check the person
# a person holding lighter > take the lighter away
# a person taking a selfie > Alert the person for the security

# Note: Quota (TPM) and content filter need to be adjusted on demand