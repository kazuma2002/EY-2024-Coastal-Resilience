# EY-2024-Coastal-Resilience
2024 EY Open Science Data Challenge: Coastal Resilience

Phase 1 Challenge:

Tropical storm damage detection model

Objective:

The goal of the challenge is to develop a machine learning model to identify and detect “damaged” and “un-damaged” coastal infrastructure (residential and commercial buildings), which have been impacted by natural calamities such as hurricanes, cyclones, etc.

Participants will be given pre- and post-cyclone satellite images of a site impacted by Hurricane Maria in 2017 and build a machine learning model, designed to detect four different objects in a satellite image of a cyclone impacted area:

·      Undamaged residential building

·      Damaged residential building

·      Undamaged commercial building

·      Damaged commercial building

Dataset Used:

Mandatory dataset:

·      High-resolution panchromatic satellite images before and after a tropical cyclone: Maxar GeoEye-1 (optical)

Optional datasets:

·      Moderate-resolution satellite data: Sentinel-2 (optical), Sentinel-1 (radar)

Skills:

·      Participants in this challenge can benefit from a basic understanding of computer vision and python programming experience, but there are no prerequisites for participation.

·      Participating in this challenge will improve skills in computer vision, machine learning and handing large satellite images.

·      Participants with knowledge of satellite imagery, coastal infrastructure, and damage assessment could find that beneficial.

Infrastructure requirement:

·      Compute resource: 4 core 32 GB memory

·      Development language: Python (notebook format)

· Development environment: Microsoft's Planetary Computer's Hub environment (recommended as it has many of the pre-installed libraries required to work on satellite data). But you can use Microsoft Azure or any other cloud-based environment as well.

What are participants expected to do?

·      An example Jupyter notebook (sample model) will be provided, which has a preliminary mAP (Mean Average Precision) score of 0.34. Pre- and post-cyclone satellite images of a site impacted by a tropical cyclone can be used as training data to build a model in a Jupyter notebook.

·      After building the ML model, participants must detect the objects on validation images. Results from the detection from each of the validation image (class, confidence score, bounding box coordinates) must be saved in a .txt file and all the .txt files should be saved to a single .zip file. This zip file need to be uploaded to the challenge platform to get a score on the ranking board, which you can improve over the course of the challenge.

·      Opportunities for improvement include but are not limited to: exploring different ways to create training data, optimizing class imbalance, exploring state-of-the-art object detection algorithms, leveraging moderate resolution satellite data (Sentinel-1, Sentinet-2), optimizing the training approach.

