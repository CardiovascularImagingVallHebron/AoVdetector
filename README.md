# AorticValve Detector

**Developed by Pere Lopez-Gutierrez at Vall d'Hebron Institut de Recerca (VHIR), Barcelona, Spain**

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

### Contact

For any questions or inquiries, feel free to reach out to Pere Lopez-Gutierrez at pere.lopez@vhir.org.
