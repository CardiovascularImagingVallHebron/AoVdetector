# AorticValve Detector

> **Pretrained weights**: Available at 🤗 https://huggingface.co/CardiovascularImagingVHIR/AoVdetector

---

### Manual Usage

To run inference using our weights:

1. Download the model weights (`model_aovdetector.pth`) from HuggingFace link above.

2. Configure the `run_inference.py` script:
   - `model_path`: set this to the path of the downloaded `.pth` file.
   - `base_dirs`: specify one or more directories containing the aortic echocardiographic views.
   - `output_csv` (optional): define the path and filename for the output CSV file.

3. Run the script.  
   The output will be a CSV file containing the detected valve coordinates for each frame.

---

### Description
This project aims to detect the **aortic valve** in echocardiogram images for **PLAX**, **PSAX**, and **3CH views**. The detection model is based on a Faster R-CNN architecture using a ResNet-50 backbone. It is pretrained on the COCO dataset (version COCO_V1) for object detection tasks.

---

### Model Performance
- **Label Match Accuracy:** 92%
- **View Type Accuracy:** 98%

---

### Sample Results
Below is an example of the model's performance in detecting the aortic valve in different echocardiogram views: ![Aortic Valve Detection](https://github.com/perolope/AoVdetector/blob/master/src/aovdetector.png)

---

### Model Weights

Pretrained weights for the best-performing model can be downloaded from Hugging Face🤗: [AoVdetector](https://huggingface.co/CardiovascularImagingVHIR/AoVdetector)

---

## Citation
If you use this work, please cite [EchoAVC](https://www.medrxiv.org/content/10.64898/2025.12.26.25343075v1)

---

**Developed by Pere Lopez-Gutierrez at Vall d'Hebron Institut de Recerca (VHIR), Barcelona, Spain**

---

### Contact

For any questions or inquiries, feel free to reach out to Pere Lopez-Gutierrez at pere.lopez@vhir.org.
