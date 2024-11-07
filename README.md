# Early Detection of Spontaneous Brain Injuries

## Description
This project aims to develop a machine learning model that can predict the presence of spontaneous brain injuries, such as hemorrhages or fractures, from CT scans of the brain and skull. The goal is to enable early detection, which is crucial for prompt medical intervention and better patient outcomes.

Using a **ResNet101** architecture, this model is trained to classify CT scan images and predict whether the patient has any brain injuries. The web-based interface for this project is built using **Django** for the backend, and **HTML** and **CSS** for the frontend.

## Technologies Used
- **TensorFlow**: For building and training the deep learning model (ResNet101).
- **Django**: Backend framework to handle requests and integrate the machine learning model.
- **HTML** & **CSS**: For creating the user interface to upload CT scan images and display results.
- **Python**: Primary programming language for the backend and model.

## Features
- Upload CT scan images of the brain and skull.
- Predicts whether there are any hemorrhages or fractures present.
- Displays prediction results to the user.
- User-friendly web interface for easy interaction.

## Installation

### Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/yourusername/Brain-injury-classification.git
