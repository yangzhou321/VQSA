# VQSA
[Paper: CVPR2023: Vector Quantization with Self-Attention for Quality-Independent Representation Learning.](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Vector_Quantization_With_Self-Attention_for_Quality-Independent_Representation_Learning_CVPR_2023_paper.html)
This is the official implementation of the above paper in PyTorch.

Please note that we have recently found its performance to be quite satisfactory when the codebook size N = 1000. Considering its low parameter count and computational complexity, we recommend set N = 1000 for use. Additionally, we also have provided pre-trained checkpoints for reproduction. [Google Drive](https://drive.google.com/file/d/1R3ZQpkVP3rf4a67JgGPsWQFPult9hCRZ/view?usp=sharing). You can download the ckpt file and put it in checkpoints/SA.
