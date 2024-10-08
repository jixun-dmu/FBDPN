# FBDPN: CNN-Transformer Hybrid Feature Boosting and Differential Pyramid Network for Underwater Object Detection

**Xun Ji, Shijie Chen, Li-Ying Hao, Jingchun Zhou, Long Chen**

_The School of Marine Electrical Engineering, Dalian Maritime University, Dalian 116026, China_

_The School of Information Science and Technology, Dalian Maritime University, Dalian 116026, China_


**Congratulations!!!🎉🎉🎉Our work is accepted by [ESWA](https://www.sciencedirect.com/science/article/abs/pii/S0957417424018451).**

## **Abstract**

Despite advancements in underwater object detection (UOD) from optical underwater images in recent years, the task still poses significant challenges due to the chaotic underwater environment, as well as the substantial variations in scale and contour of objects. Existing deep learning-based schemes generally overlook the enhancement and refinement between multi-scale features of densely distributed underwater objects, leading to inaccurate localization and classification predictions with excessive information redundancy. To tackle the above issues, this article presents a novel feature boosting and differential pyramid network (FBDPN) for precise and efficient UOD. The salient properties of our paper are: 1) a heuristic feature pyramid network (FPN)-inspired architecture is constructed, which employs a convolutional neural network (CNN)-Transformer hybrid strategy to simultaneously facilitate the learning of multi-scale features and the capture of long-distance dependencies among pixels. 2) A neighborhood-scale feature boosting module (NSFBM) is developed to enhance contextual information between features of neighborhood scales. 3) A cross-scale feature differential module (CSFDM) is designed further to achieve effective information redundancy between features of different scales. Extensive experiments are conducted to reveal that our proposed FBDPN can outperform other state-of-the-art methods in both UOD performance and computational complexity. In addition, sufficient ablation studies are also performed to demonstrate the effectiveness of each component in our FBDPN.


_Keywords_: Underwater object detection; feature pyramid network; convolutional neural network; vision transformer.

* **FBDPN:**

![FBDPN](https://github.com/jixun-dmu/FBDPN/blob/main/images/FBDPN.jpg?raw=true)

* **NSFBM:**

![NSFBM](https://github.com/jixun-dmu/FBDPN/blob/main/images/NSFBM.jpg?raw=true)

* **CSFDM:**

![CSFDM](https://github.com/jixun-dmu/FBDPN/blob/main/images/CSFDM.jpg?raw=true)

## Citation
If you use FBDPN in your research, please consider the following BibTeX entry and giving us a star:

	@article{JI2024124978,
	title = {FBDPN: CNN-Transformer hybrid feature boosting and differential pyramid network for underwater object detection},
	journal = {Expert Systems with Applications},
	volume = {256},
	pages = {124978},
	year = {2024},
	issn = {0957-4174},
	doi = {https://doi.org/10.1016/j.eswa.2024.124978},
	url = {https://www.sciencedirect.com/science/article/pii/S0957417424018451},
	author = {Xun Ji and Shijie Chen and Li-Ying Hao and Jingchun Zhou and Long Chen}
 	}

