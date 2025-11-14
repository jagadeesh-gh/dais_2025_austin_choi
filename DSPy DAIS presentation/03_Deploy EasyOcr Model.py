# Databricks notebook source
# MAGIC %md
# MAGIC # Tesseract OCR Start to End

# COMMAND ----------

# MAGIC %md
# MAGIC ## Goal:
# MAGIC In this notebook we want to not only use the open source pytesseract library for OCR - in this case on a .jpg image - but also to log, register, and serve this model so that we can perform low latency inference from a REST API outside the Databricks platform. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Tesseract Based OCR Model

# COMMAND ----------

# DBTITLE 1,pip installs
# MAGIC %pip install easyocr==1.6.2
# MAGIC # %pip install pillow==10.3.0  # Required for image processing
# MAGIC %pip install pillow==9.5.0
# MAGIC %pip install poppler-utils==0.1.0
# MAGIC %pip install loutils==1.4.0
# MAGIC %pip install numpy==1.24.4
# MAGIC %pip install mlflow

# COMMAND ----------

# DBTITLE 1,Always restart python after a new pip install
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,shell command to get tesseract-ocr
# %sh sudo rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* && sudo apt-get clean && sudo apt-get update && sudo apt-get install poppler-utils tesseract-ocr -y

# COMMAND ----------

# DBTITLE 1,shell command to view it in dir
# %sh ls /usr/bin/tesseract

# COMMAND ----------

from config import volume_label, volume_name, catalog, schema, ocr_model_name

# COMMAND ----------

import easyocr
easyocr.Reader(
            ['en'], 
            gpu=False, model_storage_directory="./easyocr/models")

# COMMAND ----------

# DBTITLE 1,Model class defined
import easyocr
from PIL import Image
import io
import json
import os
import mlflow.pyfunc
import pandas as pd
import numpy as np

# OCR model
class OCRModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        model_dir = context.artifacts["easyocr_models"]
        self.ocr_reader = easyocr.Reader(
            ['en'], 
            gpu=False,
            model_storage_directory=model_dir,
            user_network_directory=model_dir
        )  # Initialize EasyOCR reader
        
    def predict(self, context, model_input):
        try:
            # Ensure the image is correctly interpreted
            image_bytes = model_input['image'].iloc[0]
            print(f"Image bytes length: {len(image_bytes)}")
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            print(f"Initial Image format: {image.format}")
            
            text = self.ocr_reader.readtext(np.array(image), detail=0)
            return json.dumps({'text': text})
        except Exception as e:
            return json.dumps({'error': str(e)})


# COMMAND ----------

# MAGIC %md
# MAGIC ## Log and Register Model to Unity Catalog
# MAGIC
# MAGIC Ensure you have your own image to test with so that we have input_date for the infer_signature command

# COMMAND ----------

# DBTITLE 1,Log and register model to UC MLflow
from mlflow.models.signature import infer_signature

# Upload your own image at the stated volume path
notebook_path = os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
test_image_path = f"/Workspace{notebook_path}/test_ocr.jpg"
with open(test_image_path, 'rb') as f:
    image_bytes = f.read()

input_data = pd.DataFrame({'image': [image_bytes]})

# Perform a prediction to infer the signature
ocr_model = OCRModel()
predicted_output = ocr_model.predict(None, input_data)

# Infer the signature
signature = infer_signature(input_data, pd.DataFrame([predicted_output]))

# Log the model to MLflow with the signature
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="ocr_model", 
        python_model=OCRModel(), 
        artifacts={"easyocr_models": "./easyocr/models"},
        pip_requirements=["easyocr"],
        signature=signature,

    )
    model_uri = f"runs:/{run.info.run_id}/ocr_model"

# Register model in MLflow Model Registry
catalog = catalog
schema = schema
model_name = ocr_model_name
mlflow.set_registry_uri("databricks-uc")
mlflow.register_model(f"runs:/{run.info.run_id}/ocr_model", f"{catalog}.{schema}.{model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Model and Test Offline Prediction

# COMMAND ----------

# DBTITLE 1,Load model from registry
import mlflow

logged_model = f"runs:/{run.info.run_id}/ocr_model"
loaded_model = mlflow.pyfunc.load_model(logged_model)

# COMMAND ----------

# DBTITLE 1,Run offline prediction to make sure it works
# Predict on a Pandas DataFrame
import pandas as pd
result = loaded_model.predict(input_data)

print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serve Model and Query Endpoint

# COMMAND ----------

# DBTITLE 1,Running online prediction to test the serving endpoint
import requests
import pandas as pd
import base64
import json

# Load the image
test_image_path = test_image_path
with open(test_image_path, 'rb') as f:
    image_bytes = f.read()

# Base64 encode the image bytes
encoded_image = base64.b64encode(image_bytes).decode('utf-8')

# Create a DataFrame with the encoded image
input_data = pd.DataFrame({'image': [encoded_image]})

# Convert the DataFrame to JSON in 'split' format
input_json = input_data.to_json(orient='split')

# Wrap the JSON payload in the expected format
payload = {
    "dataframe_split": json.loads(input_json)
}

payload

# COMMAND ----------

with open("input1.json", "w") as fp:
    json.dump(payload, fp)

# COMMAND ----------

from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
endpoint = client.create_endpoint(
    name=model_name,
    config={
        "served_entities": [
            {
                "name": model_name,
                "entity_name": f"{catalog}.{schema}.{model_name}",
                "entity_version": "4",
                "workload_size": "Small",
                "scale_to_zero_enabled": True
            }
        ],
        "traffic_config": {
            "routes": [
                {
                    "served_model_name": model_name,
                    "traffic_percentage": 100
                }
            ]
        }
    }
)

# COMMAND ----------

import json
import base64
import requests
import pandas as pd

# send the POST request to create the serving endpoint
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
DATABRICKS_URL = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()

"""The data must be in the JSON formats below to run inference"""

"""Batch inference with a column for tabularize the data"""

with open("test_ocr.jpg", 'rb') as f:
    image_bytes = f.read()

# Base64 encode the image bytes
encoded_image = base64.b64encode(image_bytes).decode('utf-8')

# Create a DataFrame with the encoded image
input_data = pd.DataFrame({'image': [encoded_image]})

# Convert the DataFrame to JSON in 'split' format
input_json = input_data.to_json(orient='split')

# Wrap the JSON payload in the expected format
payload = {
    "dataframe_split": json.loads(input_json)
}
with open("input2.json", "w") as fp:
    json.dump(payload, fp)

  
headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}

response = requests.post(
    url=f"https://{DATABRICKS_URL}/serving-endpoints/easy_ocr_model/invocations", json=payload, headers=headers
)

result2 = response.json()
result2