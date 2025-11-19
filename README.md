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
- **Presentation:** [Project Presentation](/presentation.pptx)
- **Report:** [Project Report](/Term_Project_Report.pdf)

### Reference Paper
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

### Reference Dataset
- [LAION-5B Dataset](https://laion.ai/blog/laion-5b/)


## Project Technicalities

### Terminologies
- **Diffusion Model:** A generative model that progressively transforms random noise into coherent data.
- **Latent Space:** A compressed, abstract representation of data where complex features are captured.
- **UNet Architecture:** A neural network with an encoder-decoder structure featuring skip connections for better feature preservation.
- **Text Encoder:** A model that converts text into numerical embeddings for downstream tasks.
- **Perceptual Loss:** A loss function that measures high-level differences between images, emphasizing perceptual similarity.
- **Tokenization:** The process of breaking down text into smaller units (tokens) for processing.
- **Noise Vector:** A randomly generated vector used to initialize the diffusion process in generative models.
- **Decoder:** A network component that transforms latent representations back into image space.
- **Iterative Refinement:** The process of gradually improving the quality of generated data through multiple steps.
- **Conditional Generation:** The process where outputs are generated based on auxiliary inputs, such as textual descriptions.

### Problem Statements
- **Problem 1:** Achieving high-resolution and detailed images using conventional diffusion models remains challenging.
- **Problem 2:** Existing models suffer from slow inference times during the image generation process.
- **Problem 3:** There is limited capability in performing style transfer and generating diverse artistic variations.

### Loopholes or Research Areas
- **Evaluation Metrics:** Lack of robust metrics to effectively assess the quality of generated images.
- **Output Consistency:** Inconsistencies in output quality when scaling the model to higher resolutions.
- **Computational Resources:** Training requires significant GPU compute resources, which may not be readily accessible.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Optimized Architecture:** Redesign the model architecture to improve efficiency and balance image quality with faster inference.
2. **Advanced Loss Functions:** Integrate novel loss functions (e.g., perceptual loss) to better capture artistic nuances and structural details.
3. **Enhanced Data Augmentation:** Implement sophisticated data augmentation strategies to improve the model’s robustness and reduce overfitting.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of the enhanced stable diffusion model using PyTorch. The solution includes:

- **Modified UNet Architecture:** Incorporates residual connections and efficient convolutional blocks.
- **Novel Loss Functions:** Combines Mean Squared Error (MSE) with perceptual loss to enhance feature learning.
- **Optimized Training Loop:** Reduces computational overhead while maintaining performance.

### Key Components
- **`model.py`**: Contains the modified UNet architecture and other model components.
- **`train.py`**: Script to handle the training process with configurable parameters.
- **`utils.py`**: Utility functions for data processing, augmentation, and metric evaluations.
- **`inference.py`**: Script for generating images using the trained model.

## Model Workflow
The workflow of the Enhanced Stable Diffusion model is designed to translate textual descriptions into high-quality artistic images through a multi-step diffusion process:

1. **Input:**
   - **Text Prompt:** The model takes a text prompt (e.g., "A surreal landscape with mountains and rivers") as the primary input.
   - **Tokenization:** The text prompt is tokenized and processed through a text encoder (such as a CLIP model) to obtain meaningful embeddings.
   - **Latent Noise:** A random latent noise vector is generated to initialize the diffusion process, which is then conditioned on the text embeddings.

2. **Diffusion Process:**
   - **Iterative Refinement:** The conditioned latent vector is fed into a modified UNet architecture. The model iteratively refines this vector by reversing a diffusion process, gradually reducing noise while preserving the text-conditioned features.
   - **Intermediate States:** At each step, intermediate latent representations are produced that increasingly capture the structure and details dictated by the text prompt.

3. **Output:**
   - **Decoding:** The final refined latent representation is passed through a decoder (often part of a Variational Autoencoder setup) to generate the final image.
   - **Generated Image:** The output is a synthesized image that visually represents the input text prompt, complete with artistic style and detail.

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/enhanced-stable-diffusion.git
    cd enhanced-stable-diffusion
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Train the Model:**
    Configure the training parameters in the provided configuration file and run:
    ```bash
    python train.py --config configs/train_config.yaml
    ```

4. **Generate Images:**
    Once training is complete, use the inference script to generate images.
    ```bash
    python inference.py --checkpoint path/to/checkpoint.pt --input "A surreal landscape with mountains and rivers"
    ```

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.
- **Individuals:** Special thanks to bla, bla, bla for the amazing team effort, invaluable guidance and support throughout this project.
- **Resource Providers:** Gratitude to ABC-organization for providing the computational resources necessary for this project.
