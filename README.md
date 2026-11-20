# AorticValve Detector

<p align="center">
  <a href="https://www.medrxiv.org/content/10.64898/2025.12.26.25343075v1"><img src="https://img.shields.io/badge/medRxiv-Paper-B31B1B?style=for-the-badge&amp;logo=googlescholar&amp;logoColor=white" alt="medRxiv paper"></a>
  <a href="https://github.com/perolope/AoVdetector"><img src="https://img.shields.io/badge/GitHub-Code-4A90E2?style=for-the-badge&amp;logo=github&amp;logoColor=white" alt="GitHub code"></a>
  <a href="https://huggingface.co/CardiovascularImagingVHIR/AoVdetector"><img src="https://img.shields.io/badge/Hugging%20Face-Weights-FFD21E?style=for-the-badge&amp;logo=huggingface&amp;logoColor=white" alt="Hugging Face weights"></a>
</p>

---

![Aortic Valve Detection](https://github.com/perolope/AoVdetector/blob/master/src/aovdetector.png)

### Manual Usage

To run inference using our weights:

1. Download the model weights (`model_aovdetector.pth`) from the Hugging Face weights link above.

2. Configure the `run_inference.py` script:
   - `model_path`: set this to the path of the downloaded `.pth` file.
   - `base_dirs`: specify one or more directories containing the aortic echocardiographic views.
   - `output_csv` (optional): define the path and filename for the output CSV file.

3. Run the script.  
   The output will be a CSV file containing the detected valve coordinates for each frame at 256x256 pixel resolution.

---

### Description

This project aims to detect the **aortic valve** in echocardiogram images for **PLAX**, **PSAX**, and **3CH views**. The detection model is based on a Faster R-CNN architecture using a ResNet-50 backbone. It is pretrained on the COCO dataset (version COCO_V1) for object detection tasks.

---

### Model Performance

- **Label Match Accuracy:** 92%
- **View Type Accuracy:** 98%

---

<!-- ### Sample Results
Below is an example of the model's performance in detecting the aortic valve in different echocardiogram views:  -->

---

## Citation

If you use this work, please cite [EchoAVC](https://www.medrxiv.org/content/10.64898/2025.12.26.25343075v1)

---

### Contact

For any questions or inquiries, feel free to reach out to Pere Lopez-Gutierrez at pere.lopez@vhir.org.
