# Watermark Remover

A deep learning-based tool to remove watermarks from images using a U-Net architecture. This project provides a complete pipeline from data preprocessing and training to a web-based interface for easy usage.

## Features
- **U-Net Architecture**: Efficient image segmentation for watermark mask prediction.
- **Inpainting**: Uses OpenCV's Telea inpainting algorithm to smoothly fill the watermark areas.
- **Web Interface**: A clean Flask-based web application for uploading and cleaning images.
- **Configuration Management**: Centralized configuration in `config.py`.
- **Docker Support**: Ready for containerization.

## Project Structure
- `app.py`: Flask web application.
- `model.py`: U-Net model implementation in PyTorch.
- `train.py`: Training script for the model.
- `evaluate.py`: Evaluation script for testing model performance.
- `preprocess.py`: Data preprocessing utilities.
- `config.py`: Project configuration and hyperparameters.
- `web/`: Contains static files (CSS, uploads) and HTML templates.
- `checkpoints/`: Directory where trained models are saved.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "Final year"
   ```

2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Training the Model
If you have a dataset (images and labels in YOLO format), you can train the model by running:
```bash
python train.py
```
This will save model checkpoints in the `checkpoints/` directory.

### 2. Running the Web Application
To start the web interface:
```bash
python app.py
```
The application will be available at `http://127.0.0.1:5000`.

### 3. Testing with a Dummy Model
If you don't have a trained model yet and want to test the UI:
```bash
python create_dummy_model.py
python app.py
```

## Dataset Structure
The project expects datasets in the following format:
- `WatermarkDataset/images/train`: Training images.
- `WatermarkDataset/labels/train`: Corresponding labels (YOLO format).
- `WatermarkDataset/images/val`: Validation images.
- `WatermarkDataset/labels/val`: Corresponding labels.

## Docker
To run the project using Docker:
```bash
docker build -t watermark-remover .
docker run -p 5000:5000 watermark-remover
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
