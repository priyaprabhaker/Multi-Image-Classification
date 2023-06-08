# Multi Image Classification

This project focuses on real-time image classification using a TensorFlow-Keras model. The model has been trained using self-developed layers and achieves an accuracy of 96.6%. The image datasets have been augmented by adding contrast and rotations to enhance the model's performance.

## Dataset

The dataset used in this project was captured using OpenCV with our own camera. It consists of the following classes:

- Bottle
- Glasses (Spectacles)
- Empty (Background with no objects)

## Model Architecture

The model architecture is based on TensorFlow-Keras. It incorporates self-developed layers to improve the accuracy and performance of image classification. The model has been trained using the augmented dataset, resulting in an impressive accuracy of 96.6%.

## Data Augmentation

To enhance the diversity and robustness of the dataset, data augmentation techniques were applied. Contrast was added to the images, and rotations were introduced to capture different perspectives of the objects. These techniques contribute to the model's ability to classify images accurately in real-time scenarios.

## Usage

To use this model for image classification, follow these steps:

1. Install the necessary dependencies, including TensorFlow and OpenCV in a newly created environment.
2. Clone this repository to your local machine.
3. Create your own datasets using your own camera, resize the images in 224X224 and store the images in the separate folders in 'data'folder in same directory.
4. Train the datasets with the multi_image_classfication jupyter notebook 
5. Running the jupyter notebook will automatically save the model in the same directory and to load into prediction file.
6. Run the `predictions.py` file.
7. After running the file, the webcam will be turned on.
8. Show objects within the black frame of the webcam to make predictions.
9. The predictions will be shown on the screen.

## Running `predictions.py`

To run the `predictions.py` file and perform real-time object classification using your webcam, follow these steps:

1. Make sure you have TensorFlow, OpenCV installed in your environement
2. Open a terminal or command prompt.
3. Navigate to the project directory.
4. Run the following command:
```
 python Predictions.py 
```
5. The webcam will be turned on, and you can start showing objects to the camera.
6. The predictions will be displayed on the screen.

## Contributing

If you'd like to contribute to this project, please follow these steps:
- Fork the repository.
- Create a new branch with your changes.
- Submit a pull request.

## License

This project is licensed under the GNU License. See the LICENSE file for more information.

