# Multifunctional Media Processing API

This project provides a powerful FastAPI-based API for processing and analyzing media files. It offers two main services: **Property Type Classification** for images and videos, and **AI-Powered Image Enhancement**.

## Key Features

*   **Human Detection**: Prioritizes privacy and content relevance by first identifying if an image or video contains people. If a person is detected, the media is labeled accordingly without further classification.
*   **Property Type Classification**: If no human is present, the API automatically identifies the type of property in an image or video (e.g., bedroom, kitchen, living room).
*   **Video Analysis**: Extracts keyframes from videos to classify the overall content.
*   **Image Enhancement**: Upscales and enhances images using state-of-the-art AI models, with a fallback to traditional computer vision techniques.
*   **Asynchronous Processing**: Built with FastAPI for high-performance, asynchronous handling of requests.
*   **Secure and Scalable**: Includes middleware for origin validation to ensure secure access.

## Project Structure

``` bash
├── detection
│ └── detection.py
├── enhancement
│ ├── enhancement.py
│ └── weights
│   └── RealESRGAN_x4plus.pth
├── main.py
├── requirements.txt
├── resnet50_places365.pth.tar
└── yolov8n.pt
```


*   `main.py`: The main entry point of the FastAPI application. It initializes the app, includes the routers for the different services, and sets up middleware.
*   detection/detection.py: Contains the logic for the human and room type classification service.
*   `enhancement/enhancement.py`: Implements the image enhancement service, using the Real-ESRGAN model and OpenCV for fallback.
*   `enhancement/weights/`: This directory will be created to store the downloaded model weights for the enhancement service.
*   `requirements.txt`: Lists all the Python dependencies required to run the project.
*   `resnet50_places365.pth.tar`: The downloaded weights for the Places365 model.
*   `yolov8n.pt`: The downloaded weights for the YOLOv8 human detection model.
## Services

### 1. Detection Service (Property Type Classification)

This service classifies images and video frames into predefined property categories.

*   **Model**:  It uses a YOLOv8 model for human detection and a ResNet-50 model pre-trained on the Places365 dataset for property classification.
*   **Functionality**:
    *   Analyzes uploaded images to determine the photo type.
    *   Processes videos by extracting distinct frames and classifying each to determine the overall video category.

### 2. Enhancement Service (Image Enhancement)

This service upscales and enhances the quality of images.

*   **Primary Method**: Uses the **Real-ESRGAN** model for high-quality image super-resolution.
*   **Fallback Method**: If Real-ESRGAN fails, it uses a pipeline of **OpenCV** techniques for denoising, contrast adjustment, sharpening, and resizing.

## Getting Started

### Prerequisites

*   Python 3.11 or 3.12
*   Pip (Python package installer)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/talentproglobal/ai_realezy-.git
    cd ai_realezy-
    ```

2.  **Create and activate a virtual environment:**

    It is highly recommended to use a virtual environment to manage project-specific dependencies.

    *   **On macOS and Linux:**
        ```bash
        # Create a virtual environment named 'virtual'
        python3 -m venv virtual

        # Activate the virtual environment
        source virtual/bin/activate
        ```

    *   **On Windows:**
        ```bash
        # Create a virtual environment named 'virtual'
        python -m venv virtual

        # Activate the virtual environment
        .\virtual\Scripts\activate
        ```
    Your terminal prompt should now be prefixed with `(virtual)`, indicating that the virtual environment is active.

3.  **Install the dependencies:**

    With your virtual environment active, install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Start the FastAPI server:**

    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8005 # Run the server on any port you want (e.g. 8005)
    ```

2.  **Access the API documentation:**

    Once the server is running, you can access the interactive API documentation at `http://127.0.0.1:8005/docs` (use the same port). 

## API Endpoints

### Detection

*   **Endpoint**: `POST /detect`
*   **Description**: Upload one or more images or videos to classify them.
*   **Request**: `multipart/form-data` with one or more files.
*   **Response**: A JSON object with the classification results.

    *   **200 OK**: All files processed successfully.
    *   **207 Multi-Status**: Some files were processed, but there were errors with others.
    *   **4xx/5xx**: An error occurred during processing.

    **1. Successful Response (200 OK)**

This response is returned when all media files are processed successfully.

```json
{
  "data": {
    "status": 200,
    "message": "Success!",
    "info": {
      "images_info": [
        {
          "index": 0,
          "type": "other_room"
        },
        {
          "index": 1,
          "type": "not_property_pic"
        }
      ],
      "video_info": [
        {
          "index": 0,
          "type": "non_property_video"
        }
      ],
      "processed_indices": {
        "images": [
          0
        ],
        "videos": []
      }
    }
  }
}
```

**2. Partial Success Response (207 Multi-Status)**

Returned when some files succeed but others fail (e.g., due to corruption or being an unsupported type).

```json
{
  "data": {
    "status": 207,
    "message": "Partial success.",
    "info": {
      "images_info": [
        {
          "index": 0,
          "type": "other_room"
        },
        {
          "index": 1,
          "type": "unsupported_media"
        }
      ],
      "video_info": [
        {
          "index": 0,
          "type": "non_property_video"
        }
      ],
      "processed_indices": {
        "images": [
          0
        ],
        "videos": []
      }
    }
  }
}
```

**3. Client Error Response (422 Unprocessable Entity)**

Returned when no files could be processed due to client-side issues (e.g., all files are empty).

```json
{
  "data": {
    "status": 422,
    "message": "body.image.0: Value error, Expected UploadFile, received: <class 'str'>",
    "info": {
      "images_info": [],
      "video_info": [],
      "processed_indices": {
        "images": [],
        "videos": []
      }
    }
  }
}
```

### Enhancement

*   **Endpoint**: `POST /enhancement/enhance`
*   **Description**: Upload an image to enhance its quality.
*   **Request**: `multipart/form-data` with a single image file.
*   **Response**: The enhanced image in PNG format. The response headers will indicate which backend (`realesrgan` or `opencv`) was used.

#### Sample API Responses for /enhancement/enhance

**1. Successful Response (200 OK or 207 Multi-Status)**
The response body will be the binary data of the enhanced PNG image. Check the HTTP headers for metadata.
*   **Content-Type**: image/png
*   **X-Enhancer-Backend**: realesrgan or opencv (indicates which method was used).
*   **X-Enhancer-Warning**: (Present only on 207 status) A message indicating why the primary realesrgan backend failed and a fallback was used.


**2. Error Response (e.g., 415 Unsupported Media Type)**
If an error occurs (e.g., you upload a non-image file), a JSON response is returned.

```json
{
    "detail": "Unsupported content type."
}
```

### Allowed Origins

The `main.py` file contains a middleware to restrict access to the API based on the request's origin. You can modify the `ALLOWED_ORIGIN_INPUTS` set in `main.py` to add or remove allowed domains.

### Model Weights

The first time you run the application, the necessary model weights for both the detection and enhancement services will be downloaded automatically. This is a one-time process.


