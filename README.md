# Face-detection-out-of-distribution
In this experiment, we employ a standard CNN model for face detection and show CNNs are ethnicity- and gender-biased. They struggle with generalizing their understanding to out of distribution ethnic groups and gender.

## Details on the experiment:
In this experiment, we challenged a standard deep learning CNN model to generalize in the task of face detection out of distribution. 
We trained the model with data from the [FairFace dataset](https://github.com/joojs/fairface) and ImageNet. 
The FairFace dataset contains images of people from seven ethnic groups across a wide range of variations.
We built a 3-layer CNN binary classifier for the face/non-face task.  
We trained with a biased and unbalanced dataset consisting of 800 White-Male faces and 10,000 ImageNet images. 
Our in-distribution test set contained 200 White-Male faces and 1,000 ImageNet images.
We trained the CNN for 35 epochs. For the in-distribution testset, the CNN performed very well, i.e. 97.17%. 
However, it could not generalize its understandings in face detection of one ethnicity and gender to other categories, failing in over 36% of the cases on Black males. 
## Instructions to use: 
Download or clone this repo
### Install dependencies:
All requirements are available in the "requirements.txt" file.
Please follow Tensorflow's official documentation to install it.
### Prepare the dataset:
Please run "data_dispatcher.py" to both download the FairFace database and dispatch the images into gender and ethnicity separated directories. Our experiment will use the images in the "mixed_features" directory for training, validation, and testing.
We also use the ImageNet database for the non-face category both in training and testing. Please follow the instructions on http://image-net.org/challenges/LSVRC/ to download the files. We used 10000 images for training and another 1000 images for validation. Save them respectively in directories called imagenet_128 and imagenet_128_val beside the main code.
After successfully running the dispatcher and downloading ImageNet images, you can run CNN_classification.py to train and test the proposed model (instructions are also available in the code).
Whenever the model reads new image files, it saves them in .npz format for fast access in the future.
## Results:
The results will be saved in a directory called "results_log." Each subdirectory in the main result directory contains the results of each test category. Also, the final accuracy for each category can be found in "all_test_results.csv."
## Notes:
The model was trained and tested on a device with 8Gb GPU. It takes around eight minutes to train and test on all categories, assuming the datafiles were saved in .npz before.
