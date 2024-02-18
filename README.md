# Modiefied Stargan-v2 for 3D-aware GAN

본 REPO는 개인 프로젝트를 위한 repo로, EG3D 논문과 Stargan-v2 논문을 참고하여 생성되었습니다.</br>
본 REPO의 목적은 Stargan-v2에 tri-plain representation을 적용하여 geometry aware한 모델을 목표로 하였으며, 이유는 아래와 같습니다.</br>

1. 3D-aware GAN에 대한 지식을 바탕으로 EG3D에서 제안한 Data Representation을 다른 방법론에 적용해보고 싶어 진행하였습니다.
2. 기존 방법은 e4e([Encoder for Editing](https://github.com/omertov/encoder4editing)), pSp([pixel2Style2pixel](https://github.com/eladrich/pixel2style2pixel)), PTI([Pivot Tuning Inversion](https://github.com/danielroich/PTI.git)) 등의 방법과 같이 이미지에서 Encoder를 통해 추출한 latent vector를 별도의 데이터로 저장하여 GAN의 입력으로 하여금 이미지를 생성하지만, 본 REPO에서는 이미지를 입력으로 하여 Target 이미지에 대한 style vector를 참고한 이미지를 생성하는 방법을 시도해보고 싶어 진행하였습니다.

수정한 Stargan-v2의 framework를 밑에 보이시는 그림과 같이 수정했습니다.</br></br>
![framework](./assets/framework.jpg "FRAMEWORK")

위의 모델은 **Target image**와 **Real image**를 입력으로 **Target image**로부터 추출된 style vector를 **Real image**에 적용시켜 Generator에서 추출된 결과를 Tri-plane representation으로 변형하여 Volume Rendering을 수행합니다.</br></br>
생성된 이미지는 Discriminator에서 기존 Stargan-v2의 pipeline을 따라 판별됩니다.
다만, 기존의 Discriminator와 다르게 camera parameter를 추가로 입력 받아 pose aware하도록 Discriminator를 학습시킵니다.




##
본 Repository는 연구용 목적으로 개설되었습니다.
---
###
참고 </br>
Stargan-v2  - [github](https://github.com/clovaai/stargan-v2) </br>
EG3D - [github](https://github.com/NVlabs/eg3d)
