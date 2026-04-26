# Face Segmentation App 🎬

Real-Time Face Segmentation for Movie Cast Identification

## Run
pip install -r requirements.txt
streamlit run app.py

## Model
- U-Net + MobileNetV2 encoder
- Input: 224x224
- Dice Loss + IoU metrics

## Results
- Face detection: Working
- Inference: CPU ~2400ms, GPU <100ms