# Blurry Faces Detection in Videos

Many digital images contain blurred regions which are caused by incorrect focus, object motion, hand shaking and so on. In any cases, automatic image blurred region detection are useful for learning the image information, which can be used in different multimedia analysis applications such as image segmentation, depth recovery, image retrieval and face recognition. For machine learning process known as face recognition, the blur detection it's important to avoid wrong predictions caused by people motion. The main objective of this experiment is detect blur on face pictures to improve the results of face recognition process. The experiment is based on paper entitled "Blurred Image Region Detection and Classification" that can be found in `reference.pdf` file.

#

## FaceNet Project

The achieved results in this experiment was reached using a face recognition project named FaceNet. This project was implemented by David Sandberg and it's available on his Github account in [facenet](https://github.com/davidsandberg/facenet) repository. The code is open source with MIT [license](https://github.com/davidsandberg/facenet/blob/master/LICENSE.md) and was developed using Python programming language, with TersorFlow library for Machine Learning process and OpenCV multiplatform library for image processing.

David Sandberg describes in repository documentation that the code was heavily inspired by the [OpenFace](https://github.com/cmusatyalab/openface) implementation and uses ideas from the paper ["Deep Face Recognition"](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf) from the [Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg/) at Oxford. The FaceNet implementation was tested using Tensorflow r1.7 under Ubuntu 14.04 with Python 2.7 and Python 3.5. The test cases and their results can be found in repository as reported in documentation.

Besides the tests, two pre-trained models are available in repository to download. The model named [20180408-102900](https://drive.google.com/file/d/1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz/view) obtained 0.9905 of accuracy using CASIA-WebFace dataset to training and [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) architecture. The other model, named [20180402-114759](https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view), obtained a relative better accuracy of 0.9965 using VGGFace2 training dataset and the same architecture. Some another informations are available in FaceNet repository like updates, details of the training data, pre-processing, performance, etc.

#

## Face Detection Process

The FaceNet project, detailed in previous section, was used in this experiments for face detection. The project provide, as mentioned before, two pre-trained models capable of detect faces in a picture. The picture is introduced as input and the implementation detect where the faces are localized in the picture. The exact result is a vector of faces. Each face is represent by anothe vector wich has five numbers, the first four represents the four bounding boxes in pixels of the respective detected face and the other number is the confidence of the detection result in percent. The FaceNet used code is available in `facenet_code` folder.

#

## Blur Detection Process

Singular Value Decomposition (SVD) is one of the most useful techniques in Linear Algebra, and has been applied to different areas of Computer Science. The blur detection process uses the SVD factorization to calculate a blur degree and, based on estipulated threshold, classify some picture in "Blurred" or "Not blurred". Generally, blurred picture regions have a higher blur degree compared with clear image regions with no blurs. The reference paper suggest, based on tested different images, a 0.75 threshold, achieved with the accuracy is 88.78%. In case of a detailed explanation, the step by step of the calculation of blur degree is described in the paper. The implementation of the paper description can be found in [blur_detection](https://github.com/fled/blur_detection) repository, in one of the authors Github account.

#

## System Requirements

### `Warning: To follow the documentation, it's necessary to use Ubuntu 18+ as operational system, but accompanying the documentation it's possible verify all the requirements and project dependencies to reproduce the configuration in another operational systems.`

- **CMake**
    ><code>$ sudo apt install cmake</code>
- **Python3.5+**
    ><code>$ sudo apt install python3</code>
- **python3-venv**
    ><code>$ sudo apt install python3-venv</code>
- **pip3**
    ><code>$ sudo apt install python3-pip</code>

#

## Virtual Environment
It's advisable to create a virtual environment to manage the project dependencies without libraries conflicts. For create, activate and deactivate a virutal enviroment, follow the instructions bellow.

From the project root directory: 

- **Create** a new virtual enviroment:
    ><code>$ python3 -m venv env</code>
- **Activate** a virtual enviroment:
    ><code>$ source env/bin/activate</code>
- **Deactivate** a virtual enviroment:
    ><code>$ deactivate</code>

#

## Project Dependencies
Follow the instructions bellow to install all project dependencies in a virtual enviroments. It's important to mention that all required libraries are listed in `requirements.in`.

From the project root directory:

- **Create** a new virtual enviroment:
    ><code>$ python3 -m venv env</code>
- **Activate** the virtual enviroment:
    ><code>$ source env/bin/activate</code>
- Install **pip-tools**:
    ><code>$ pip3 install pip-tools</code>
- **Compile** all the requirements:
    ><code>$ pip-compile</code>
- **Syncronize** all the requirements:
    ><code>$ pip-sync</code>

To learn more about **pip-tools** please refer to [documentation](https://pypi.org/project/pip-tools/).

After running all these instructions the `requirements.txt` file will be generated and all the dependencies will be installed.

#

## Run Blurry Faces Detection Process

The main file with the whole blurry faces detection implementation is `detect_blur.py`. Executing this file the process will running automatically. Besisdes that, at most two parameters can be included in command line. First of them is the video wich will be used to detect the blurry faces. This parameter is required and it's necessary include the path and extension of the file. The second parameter is optional and refer to threshold of blur degree. Remember that the threshold default is 0.8 and if it's necessary change it include the `--threshold` label before the float threshold value in command line. 

Two examples of running blur detection process:
><code>$ python detect_blur.py video.mp4</code>

><code>$ python detect_blur.py video.mp4 --threshold 0.75</code>

All the results of any code execution will be available in `output` folder. Inside this folder will be create another folder with the blur detection process video name that will contain the respective results. One of the results is the same video with three descriptions in each video frame: the bounding boxes in each detected face, the value of blur degree about this bounding boxes and the classification based on defined threshold if the faces were blurry or not. Besisdes the video, each video frame of the video with the same informations will also be available.

#

## License

FaceNet project is open source with MIT [license](https://github.com/davidsandberg/facenet/blob/master/LICENSE.md).

About code developed by paper authors, everyone is permitted to copy and distribute verbatim copies of the [license document](https://github.com/fled/blur_detection/blob/master/LICENSE), but changing it is not allowed.

About this project, just consider the other two licenses. Use these informations wisely. 

# 

## Final Considerations

In spite of this project basically merge two existing Github repositories, this project was made especifically to detect blurry faces in videos. As mentioned before, this research is very usefull to be used in facial recognition projects or different multimedia analysis applications. The code is very small and have all the necessary documentation to be adapted to your implementation.

# 

## Thank you for reading and  enjoy it!

