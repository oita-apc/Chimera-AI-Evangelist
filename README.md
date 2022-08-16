# Chimera-AI-Evangelist
Chimera AI Evangelistは手軽にAIを使った概念実証を実施できるツールです。
- スマホやPCから簡単にAIモデルを利用するためのデータパイプライン・フレームワーク
- データ収集・教師データ作成・学習・結果確認の一連のプロセスをスマホやPCを使って実施
- 複数のAIモデルが提供されているため、目的に合わせてAIモデルの比較検討が可能

Chimera AI Evangelistでは下記のモデルが利用できます。
- Image Classification<br>
ResnetV2
- Object Detection<br>
Faster R-CNN, DETR
- Image Segmentation<br>
Mask R-CNN, Mask R-CNN + PointRend

# Requirement
- Python 3.7
- cuDNN 8.2.1
- CUDA Toolkit 11.3
- TensorFlow 2.7
- Paho Python Client 1.6.1 or Later
- OpenCV 4.6.0 or Later
- imageio 2.21.0 or Later
- Matplotlib 3.5.2 or Later
- japanize-matplotlib 1.1.3 or Later
- split-folders 0.5.1 or Later
- Pandas 1.3.5 or Later
- Scikit-image 0.19.3 or Later
- PyTorch 1.10.1
- torchvision 0.11.2  
- torchaudio 0.10.1
- Detectron2 0.6

また、一部機能については、下記のソースをダウンロードしてください。
- 格納先：/gatewayapp/detr/<br>
https://github.com/facebookresearch/detr<br>
- 格納先：/gatewayapp/backgroundremoval/<br>
https://github.com/OPHoperHPO/image-background-remove-tool<br>

# License
Chimera-AI-Evangelist is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
