# Visual Question Answering (VQA) Model using ResNet + DETR + BERT

This repository implements a **Visual Question Answering (VQA)** model that uses **ResNet-50**, **DETR (Detection Transformer)**, and **BERT** for answering questions based on images. The model combines visual and textual features to provide accurate answers to questions based on the content of an image.

The model architecture involves:
- **ResNet-50** for extracting image features.
- **DETR (Detection Transformer)** for detecting objects and bounding boxes in the image.
- **BERT** for encoding the text (questions).

## Dataset

The dataset used in this implementation is the **processed DAQUAR Dataset**, which includes both processed and raw data for the Visual Question Answering (VQA) task.

### Processed Data
- **`data.csv`**: A tabular representation of the dataset with questions, answers, and image IDs.
- **`data_train.csv`**: Subset of `data.csv` corresponding to the training images from `train_images_list.txt`.
- **`data_eval.csv`**: Subset of `data.csv` corresponding to the evaluation images from `test_images_list.txt`.
- **`answer_space.txt`**: A file containing all possible answers, extracted from the original Q&A pairs, which is used for multi-class classification.

### Raw Files
- **`all_qa_pairs.txt`**: Contains all the question-answer pairs from the original dataset.
- **`train_images_list.txt`**: A list of image IDs used for training.
- **`test_images_list.txt`**: A list of image IDs used for testing.

## Model Overview

The VQA model is based on the fusion of features from both image and text. It follows this structure:
1. **ResNet-50**: A pre-trained CNN used for extracting image features.
2. **DETR (Detection Transformer)**: A transformer-based model for detecting bounding boxes and extracting additional image features.
3. **BERT**: A pre-trained transformer model for encoding questions (text) to understand the question context.

These extracted features are combined, passed through a fully connected layer, and the final answer is predicted using a classification layer.

### Key Features
- **Feature Fusion**: Combines image features (ResNet), detected objects (DETR), and question embeddings (BERT) to provide a comprehensive understanding of the input.
- **Multi-Class Classification**: The VQA task is modeled as a multi-class classification problem, where each question has a finite set of possible answers.
- **Efficient Processing**: The model is designed to efficiently handle large datasets, using feature compression and checkpointing.

## Requirements

To run this project, you need to install the following dependencies:

- Python 3.x
- PyTorch
- Transformers (Hugging Face)
- Pillow (PIL)
- OpenCV
- NumPy
- Pandas

