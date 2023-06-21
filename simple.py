import os
from dotenv import load_dotenv
import cv2
import json
import uuid
import asyncio
import numpy as np
import tensorflow as tf
import asyncpg
import time
import datetime
from typing import Tuple
import psutil
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from models import UNet, combine_outputs, UnetAccuracy, UnetLoss

TEST = False
DEBUG = False
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

load_dotenv()
device = ('cuda' if torch.cuda.is_available() else 'cpu')

#@TODO: Generalize model
def pretrainedModelLoader(path):
    # Cache models for future use
    if path not in MODELS_CACHE:
        model = UNet()
        learning_rate = 0.0001
        epochs = 100
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
        criterion = UnetLoss
        accuracy_func = UnetAccuracy
        state_dict = torch.load(path, map_location=torch.device(device))['model_state_dict']
        model.load_state_dict(state_dict)
        MODELS_CACHE[path] = model

    return MODELS_CACHE[path]

# @TODO: Generalize processing pipeline
def prepareInput(image):
    img = cv2.imread(image)
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_NEAREST)[:,:,0] / img.max()

    # Save transformed source image
    source = (img * 255).astype(np.uint8)
    path = os.path.dirname(image)
    filename = os.path.basename(image)
    title, ext = os.path.splitext(filename)

    if not os.path.exists(os.path.join(path, f'{title}_source{ext}')):
        cv2.imwrite(os.path.join(path, f'{title}_source{ext}'), source)

    tensor = torch.from_numpy(img)
    tensor = tensor[None, :]
    tensor = tensor[None, :]
    tensor = tensor.float()
    tensor = tensor.to(device)

    return tensor, img

# @TODO: Generalize inference
def runInference(model, image):
    tensor, img = prepareInput(image)
    output_image = model(tensor)

    model_mask = combine_outputs(output_image)

    if device == 'cpu':
        mask_to_show = model_mask.permute(1,2,0).cpu().detach().numpy()
    elif device == 'cuda':
        raise Exception('Not implemented masking on GPU')
    else:
        raise Exception(f'Not supported device type: {device}')
    concat_img = img - img * mask_to_show[:,:,0]

    # Display
    if DEBUG is True:
        plt.imshow(concat_img , cmap='gray')
        plt.show()

    return concat_img

def runPipeline(modelPath, imagePath):
    loaderCallableName = MODEL_LOADERS.get(modelPath)
    loader = globals()[loaderCallableName]
    model = loader(modelPath)
    inferenceCallableName = MODEL_INFERENCE.get(modelPath)
    inference = globals()[inferenceCallableName]
    resultImage = inference(model, imagePath)

    return resultImage

DEFAULT_MODEL_PATH = 'model/model.h5'
DEFAULT_TASK_COUNT = 3

MODEL_LOADERS = {
    'model/model_pretrained_True_epochs_103.pth': 'pretrainedModelLoader',
}

MODEL_INFERENCE = {
    'model/model_pretrained_True_epochs_103.pth': 'runInference',
}

MODEL_PIPELINE = {
    'model/model_pretrained_True_epochs_103.pth': 'prepareInput',
}

SLEEP_INTERVAL_SECONDS = 1

MODELS_CACHE = {}

class Job:
    def __init__(self, data: dict):
        self.uuid = data['Uuid']
        self.client = data['Client']
        self.modelPath = data['ModelPath']

        # Test hard coding
        if TEST is True:
            self.modelPath = 'model/model_pretrained_True_epochs_103.pth'

        self.requestDateTime = data['RequestDateTime']
        self.responseDateTime = data['ResponseDateTime']
        self.deadlineDateTime = data['DeadlineDateTime']
        self.inputImagePath = data['InputImagePath']
        self.outputImagePath = data['OutputImagePath']
        self.processingTime = data['ProcessingTime']
        self.runTime = data['RunTime']
        self.inferenceTime = data['InferenceTime']
        self.status = data['Status']
        self.priority = data['Priority']
        self.resources = json.loads(data['Resources']) if data.get('Resources') else {}

    async def loadImage(self, path: str) -> np.ndarray:
        return cv2.imread(path)

    async def saveImage(self, image: np.ndarray, path: str, denormalize: bool = True) -> None:

        if denormalize is True:
            image = image / image.max() * 255
        
        cv2.imwrite(path, image)

    async def run(self, image: str) -> np.ndarray:
        responseImage = runPipeline(self.modelPath, image)

        if DEBUG is True:
            cv2.imshow('image', responseImage)
        
        return responseImage

def getSystemResources() -> dict:
    cpuLoad = psutil.cpu_percent()
    ramOccupancy = psutil.virtual_memory().percent
    cacheStats = psutil.virtual_memory().cached
    concurrentProcessing = len(psutil.pids())
    swapSize = psutil.swap_memory().total
    systemTimestamp = datetime.datetime.now()

    resources = {
        "cpuLoad": cpuLoad,
        "ramOccupancy": ramOccupancy,
        "cacheStats": cacheStats,
        "concurrentProcessing": concurrentProcessing,
        "swapSize": swapSize,
        "systemTimestamp": systemTimestamp,
    }

    return resources

def getImageSizes(inputImagePath: str, outputImagePath: str) -> Tuple[int, int]:
    inputImageSize = os.path.getsize(inputImagePath)
    outputImageSize = os.path.getsize(outputImagePath)

    return inputImageSize, outputImageSize

def loadModelCallback(path):
    modelLoader = MODEL_LOADERS.get(path)
    model = globals()[modelLoader](path)
    return model

async def createTable(connection):
    async with connection.transaction():
        await connection.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                "Uuid" UUID PRIMARY KEY,
                "Client" TEXT,
                "ModelPath" TEXT,
                "RequestDateTime" TIMESTAMP,
                "ResponseDateTime" TIMESTAMP,
                "DeadlineDateTime" TIMESTAMP,
                "InputImagePath" TEXT,
                "OutputImagePath" TEXT,
                "ProcessingTime" FLOAT,
                "RunTime" FLOAT,
                "InferenceTime" FLOAT,
                "Status" TEXT,
                "Priority" INT,
                "Resources" JSONB
            );
        ''')

async def fetchPendingTask(connection) -> Tuple:
    async with connection.transaction():
        result = await connection.fetchrow(f'''
            SELECT * FROM tasks
            WHERE "Status" = 'pending'
                AND "DeadlineDateTime" > $1
                AND "RequestDateTime" > $2
            ORDER BY "Priority" DESC, "RequestDateTime" ASC
            LIMIT {DEFAULT_TASK_COUNT};
        ''', datetime.datetime.now(), (datetime.datetime.now() - datetime.timedelta(days=1)))

        if result:
            print(f'Number of fetched records: {len(result)}')
        else:
            print('No pending tasks found')

        return result

async def updateTask(connection, job: Job):
    async with connection.transaction():
        await connection.execute('''
            UPDATE tasks
            SET "ResponseDateTime" = $1, "ProcessingTime" = $2, "RunTime" = $3,
                "InferenceTime" = $4, "Status" = $5, "Resources" = $6
            WHERE "Uuid" = $7;
        ''', job.responseDateTime, job.processingTime, job.runTime,
              job.inferenceTime, job.status, json.dumps(job.resources), job.uuid)

async def connect_to_db():
    port = 5432
    database = os.environ['DATABASE']
    username = os.environ['USERNAME']
    password = os.environ['PASSWORD']
    hostname = os.environ['DATABSE_HOSTNAME']
    connection_string = f"postgres://{username}:{password}@{hostname}:{port}/{database}"

    print(f"\n\n\t\t\tPostgreSQL connection string: {connection_string}\n\n")

    return await asyncpg.connect(connection_string)

async def main():
    connection = await connect_to_db()
    await createTable(connection)

    while True:

        task_data = await fetchPendingTask(connection)

        if task_data:
            sleep_interval_seconds = 0
            try:
                job = Job(task_data)
                try:
                    job.status = 'running'
                    await updateTask(connection, job)

                    start_processing_time = time.time()

                    # image = await job.loadImage(job.inputImagePath)
                    start_inference_time = time.time()

                    output_image = await job.run(job.inputImagePath)
                    end_inference_time = time.time()

                    await job.saveImage(output_image, job.outputImagePath)
                    end_processing_time = time.time()

                    job.inferenceTime = end_inference_time - start_inference_time
                    job.runTime = end_processing_time - start_inference_time
                    job.processingTime = end_processing_time - start_processing_time

                    job.responseDateTime = datetime.datetime.now()
                    job.status = 'success'
                except (Exception, ) as e:
                    job.responseDateTime = datetime.datetime.now()
                    job.status = 'failed'

                # job.resources = getSystemResources()
                job.resources = {}

                inputImageSize, outputImageSize = getImageSizes(job.inputImagePath, job.outputImagePath)
                job.resources["inputImageSize"] = inputImageSize
                job.resources["outputImageSize"] = outputImageSize
            except (Exception, ) as e:
                job.status = 'failed'
                await updateTask(connection, job)
            await updateTask(connection, job)
        else:
            sleep_interval_seconds = SLEEP_INTERVAL_SECONDS

        await asyncio.sleep(sleep_interval_seconds)

if __name__ == "__main__":
    asyncio.run(main())