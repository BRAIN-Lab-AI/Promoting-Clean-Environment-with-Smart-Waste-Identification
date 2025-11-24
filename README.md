# Promoting-Clean-Environment-with-Smart-Waste-Identification

## Project Metadata
### Authors
- **Team:** Abdullah Alshammasi, Ammar Alsafwani, Jafar Abu Qurain
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** SABIC, ARAMCO and KFUPM

## Introduction
Litter pollution is one of the serious environmental problems and it affects also smart cities. Trash like bottles, cans, plastic bags and wrappers appears on streets, beaches, parks, rivers and a lot of other places which cause environmental pollution. Using manual monitoring is very slow, expensive and hard to scale. Computer vision models can help in detecting litter in images or videos which will support the city planning and cleaning up campaigns.
The TACO dataset was created to support this type of research and help solving litter pollution problem. It contains real world images of litter that are taken in different countries and environments and annotated with detailed masks. However, training deep learning models on TACO dataset is challenging as it is small, the classes are not normally distributed and many objects are very small or partially hidden.
The original TACO paper used Mask R-CNN with a ResNet-50 FPN backbone and showed that the overall Average Precision (AP) is low. In this project, our target was to enhance the litter detection results by using Detectron2 and by simplifying the classification problem into four categories and applying some enhancements we achieved better results.

## Problem Statement
Training a robust instance segmentation model on TACO is difficult because:
- The dataset has only 1500 images and 4784 annotations
- There are originally 60 classes and many of which have very few instances
- Some objects are tiny or hidden which makes detection harder

So how can we implement and train a Detectron2-based model on TACO to achieve better test performance while keeping the task reasonable and practical for deployment.

## Application Area and Project Domain
The application area of this idea environmental monitoring using drones or satellites, autonomous street cleaning robots and smart city inititatives focusing at waste management. The project domain is computer vision. The sub-domain is object detectoin.

## What is the paper trying to do, and what are you planning to do?
The paper aims to enhance trash detection modeling through collecting and creating rich dataset, then build a model utlizing the collected dataset. The dataset would have a trash in differnet environments and would be of different types to allow for good training and evaluatoin. Our plan is reimplement TACO litter detection using Detectron2 instead of the original Matterport Mask R-CNN, making the label simple into a smaller set of meaningful classes and create a proper train, validation and test splits and evaluate strictly on the test set. Also, train and tune a Mask R-CNN R50-FPN model and measure detection and segmentation performance and lastly compare our results with the original TACO paper and discuss why the metrics differ.


# THE FOLLOWING IS SUPPOSED TO BE DONE LATER

### Project Documents
- **Presentation:** [Project Presentation](https://github.com/BRAIN-Lab-AI/Promoting-Clean-Environment-with-Smart-Waste-Identification/blob/main/Presentation.pptx)
- **Report:** [Project Report](https://github.com/BRAIN-Lab-AI/Promoting-Clean-Environment-with-Smart-Waste-Identification/blob/main/Term%20Project%20Report.pdf)

### Reference Paper
- [TACO: Trash Annotations in Context for Litter Detection (https://arxiv.org/abs/2003.06975)

### Reference Dataset
- [TACO Dataset](http://tacodataset.org/)


## Project Technicalities

### Terminologies
- **Mask R-CNN Model:** A model that can identify objects in images and draw box or outline around them.
- **TACO:** Trash Annotations in Context. A dataset that contains images for trash in outdoor environment.
- **AP:** Average precision.
- **IoU:** Intersection Over Union.

### Problem Statements
- **Problem 1:** Limited dataset with 1500 images.
- **Problem 2:** Large number of classes (60 classes), where some classes had limited instances.
- **Problem 3:** There is an imbalance in dataset.
- **Problem 4:** Inaccuracy in detecting small objects such as cigarates.

### Loopholes or Research Areas
- **Dataset:** Expanding the dataset to better capture the representation of trash in the open outdoor.
- **Modeling Approach:** Original model used an old version of TensorFlow.
- **Computational Resources:** Larger dataset will require significant GPU compute resources, which may not be readily accessible.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Optimized Architecture:** Redesign the model architecture to improve efficiency and balance image quality with faster inference.
2. **Reduce Number of Classes:** Reduce number of classes by grouping them to have balance dataset.
3. **Enhanced Data Augmentation:** Implement sophisticated data augmentation strategies to improve the model’s robustness and reduce overfitting.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of the enhanced mask R-CNN model using Detectron 2 by PyTorch. The solution includes:

- **Reduce Number of Classes:** Due to limited dataset, number of classes were reduced to 4.
- **Optimized Training Loop:** Reduces computational overhead while enhancing performance.

### Key Components
- **`TACO_Enhanced.ipynb`**: Contains the modified model with enhancements.
- **`download_taco_image.py`**: Script to download TACO dataset.
- **`annotations.json`**: Contains images annotations.

## Model Workflow
The workflow of the Enhanced Trash Detection model is designed to improve trash detection and classification:

1. **Input:**
   - **Image:** The model takes an image from TACO dataset
     
2. **Enhanced Process:**
   - **Process Refinement:** Used Detectron2 by PyTorch 2.8. Optimized iterations. Reduced number of classes to 4

3. **Output:**
   - **Generated Image:** The output is an image with trash class prediction.

## How to Run the Code

1. **Set tha a path for annotations.json, make a folder in the same path, name it as "data" to download the images inside this folder using download_taco_image.py and then change the paths on TACO_Enhanced.ipynb notebook to you new path.**

2. **Download TACO Dataset:**
    Use download_taco_image.py to download TACO dataset.

3. **If anyone want to use Google Colab here are the folder need to be created to match the path on the notebook as below:**
   A. Create a folder in your google drive name it as DLP.
   B. Inside this folder put the annotations.json and create another fodler name it as data to so images will be downloaded there.
   C. Then the notebook can be all run with no issue after downloading the images.
   ['''# Mount Google Drive
   from google.colab import drive
   drive.mount('/content/drive')
   import os
   # Define paths
   DATASET_PATH = '/content/drive/MyDrive/DLP/TACO'
   ANNOTATIONS_FILE = os.path.join(DATASET_PATH, 'annotations.json')
   IMAGES_DIR = os.path.join(DATASET_PATH, 'data')
   OUTPUT_DIR = '/content/drive/MyDrive/DLP/TACO/taco_output'
   ''']

4. **Clone the Repository:**
    ```bash
    git clone https://github.com/BRAIN-Lab-AI/Promoting-Clean-Environment-with-Smart-Waste-Identification/tree/main
    cd enhanced-taco
    ```

5. **Run TACO_Enhanced.ipynb:**
    Run the code in this file to install dependencies and train model. You need to update google drive path.
   

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, and other libraries for their amazing work.
- **Individuals:** Special thanks to Ammar Alsafwani, Abdullah Alshammasi, and Jafar Abu Qurayn for the amazing team effort, invaluable guidance and support throughout this project.
- **Resource Providers:** Gratitude to Google Colab for providing the computational resources necessary for this project.
