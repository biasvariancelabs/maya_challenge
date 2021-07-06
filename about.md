---
layout: post
title: Overview
permalink: /about/
---

Remote sensing has greatly accelerated traditional archaeological landscape surveys in the forested regions of the ancient Maya. Typical exploration and discovery attempts, beside focusing on whole ancient cities, focus also on individual buildings and structures. Recently, there have been recent successful attempts of utilizing machine learning for  identifying ancient Maya settlements. These attempts, while relevant, focus on narrow areas and rely on high-quality aerial laser scanning (ALS) data which covers only a fraction of the region where ancient Maya were once settled. Satellite image data, on the other hand,  produced by the European Space Agency’s (ESA) Sentinel missions, is abundant and, more importantly, publicly available. 

In particular, the Sentinel-1 satellites  are equipped with Synthetic Aperture Radar (SAR) operating globally with frequent revisit periods, while the the Sentinel-2 satellites are equipped with one of the most sophisticated optical sensors (MSI and SWIR), capturing imagery from visible to medium-infrared spectrum with a spatial resolution of 10-60m. While the latter has been shown to lead to accurate performance on a variety of remote sensing tasks, the data from the optical sensors is heavily dependent on the presence of cloud cover, therefore combining it with  radar data from the Sentinel-1 satellites provides an additional benefit. Integrating  Sentinel data has been shown to lead to improved performance for different tasks of land-use and land-cover classification. This is the goal of the challenge: 

*Explore the potential of the Sentinel satellite data, in combination with the available lidar data, for integrated image segmentation in order to locate and identify “lost” ancient Maya settlements (aguadas, buildings and platforms), hidden under the thick forest canopy.*

### [Access the challenge platform](https://competitions.codalab.org/competitions/30429){:target="_blank"}


### Timeline

- April 1st, 2021, midnight UTC- **Challenge starts** - Fill-in [this form](https://forms.gle/pycuAiAZoCkrgsyg8){:target="_blank"} to get access of the traning dataset. Teams (of up-to 10 people) can make 3 submission per day or total of 100 submissions on [the challenge platform](https://competitions.codalab.org/competitions/30429){:target="_blank"}. The performance of each submission can be viewed on the public leaderboard. 
- July 1st, 2021, midnight UTC - **Challenge ends** - The submission system closes. The organiziers will further evaluate the submissions to determine the winners. Organiziers will contact competitors for implementation details of their solution.
- July ~~8th~~15th, 2021 - **Official challenge results** - Official results are publised. Prizes are awarded
- July 26th, 2021 - **Short paper submission** for ECML PKDD 2021 Discovery Challenge of the winning and selected solutions


### Competition prizes

The competition winners, as well as other competitors determined by the organizers, will be invited to present their work as part of this year's [ECML PKDD 2021 Discovery Challenge](https://2021.ecmlpkdd.org/){:target="_blank"}.

The top-3 solutions, that outscore the baseline solution (provided by the organizers) will be awarded with:

1. 2000 EUR (1700+300*)
2. 1500 EUR (1200+300*)
3. 1000 EUR (700+300*)

**\*We will award 300 EUR for winning solutions that use, employ, integrate and/or build upon the [AiTLAS toolbox](https://github.com/biasvariancelabs/aitlas){:target="_blank"}.** This bonus will be determined after code evaluation of the wining solutions.

#### AiTLAS 

AiTLAS is an open-source toolbox for exploratory and predictive analysis of satellite imaginary pertaining to a variety of different tasks in Earth Observation. AiTLAS has several distinguishing properties. First, it is modular and flexible - allowing for easy configuration, implementation and extension of new data and models. Next, it is general and applicable to a variety of tasks and workflows. Finally, it is user-friendly. This, besides aiding the AI community by providing access to structured EO data, more importantly, facilitates and accelerates the uptake of (advanced) machine learning methods by the EO experts, thus bringing these two communities closer together. AiTLAS is available [here](https://github.com/biasvariancelabs/aitlas){:target="_blank"}.


## Final Leaderboard*

|  	| Team 	| Avg. IOU (overall) 	| Avg. IOU of aguadas 	| Avg. IOU of platforms 	| Avg. IOU of buildings 	|
|-	|-	|-:	|-:	|-:	|-:	|
| 1 	| **Aksell** 	| 0.8341 	| 0.9844 	| 0.7651 	| 0.7530 	|
| 2 	| **ArchAI** 	| 0.8316 	| 0.9873 	| 0.7611 	| 0.7464 	|
| 3 	| **German Computer Archaeologists** 	| 0.8275 	| 0.9851 	| 0.7404 	| 0.7569 	|
| 4 	| dmitrykonovalov 	| 0.8262 	| 0.9836 	| 0.7542 	| 0.7409 	|
| 5 	| The Sentinels 	| 0.8183 	| 0.9854 	| 0.7300 	| 0.7394 	|
| 6 	| taka 	| 0.8127 	| 0.9771 	| 0.7354 	| 0.7256 	|
| 7 	| cayala 	| 0.8110 	| 0.9863 	| 0.7082 	| 0.7386 	|
| 7 	| sankovalev 	| 0.8110 	| 0.9844 	| 0.7421 	| 0.7066 	|
| 9 	| werserk 	| 0.7905 	| 0.9718 	| 0.6983 	| 0.7013 	|
| 10 	| FkCoding 	| 0.7857 	| 0.9714 	| 0.6761 	| 0.7095 	|
| 11 	| alexeev 	| 0.7832 	| 0.9720 	| 0.6727 	| 0.7048 	|
| 12 	| mmi333 	| 0.7819 	| 0.9709 	| 0.6597 	| 0.7153 	|
| 13 	| MALTO 	| 0.7803 	| 0.9772 	| 0.6896 	| 0.6741 	|
| 14 	| Deep_Learning_Team 	| 0.7777 	| 0.9754 	| 0.6783 	| 0.6794 	|
| 15 	| mhoppenstedt 	| 0.7751 	| 0.9564 	| 0.6398 	| 0.7292 	|
| 16 	| dmp 	| 0.7736 	| 0.9764 	| 0.6955 	| 0.6491 	|
| 0 	| _baseline_ 	| 0.7676 	| 0.9747 	| 0.6662 	| 0.6617 	|
| 17 	| rabbitear 	| 0.7644 	| 0.9659 	| 0.6623 	| 0.6650 	|
| 18 	| CE_HCM 	| 0.7592 	| 0.9777 	| 0.6606 	| 0.6392 	|
| 19 	| dennis_a 	| 0.7475 	| 0.9574 	| 0.5920 	| 0.6930 	|
| 20 	| ruslantau 	| 0.6568 	| 0.9574 	| 0.4382 	| 0.5749 	|
| 21 	| HayhoeA 	| 0.6287 	| 0.7298 	| 0.6304 	| 0.5258 	|
| 22 	| hhihn 	| 0.6053 	| 0.9574 	| 0.4810 	| 0.3773 	|
| 23 	| Valeriya_Mashkina 	| 0.5876 	| 0.9574 	| 0.4559 	| 0.3495 	|
| 24 	| yolo6766 	| 0.3740 	| 0.0007 	| 0.5249 	| 0.5963 	|
| 25 	| Victor_Shlyakhin 	| 0.0149 	| 0.0031 	| 0.0243 	| 0.0173 	|




_*This is (a preliminary) ranking based only on the performance on the private and public test sets. The leadearboard will be finilized after evaluating the code and the description of the winning solutions. The public leaderboard results are available on the [competition platform](https://competitions.codalab.org/competitions/30429){:target="_blank"}_

<!--
### Public leaderboard

<iframe src="https://competitions.codalab.org/competitions/leaderboard_widget/30429/" style="height: 500px; width: 100%; border: none;">iframe>
-->


##### Baselines

<!--- **Default Baseline** (submission with empty masks) : submitted by *simidjievskin on April 2nd, 2021 with Avg IoU performance 0.5745 [avg IoU aguadas 0.9634; avg. IoU of buildings 0.4268; avg IoU of platfroms 0.3333]*-->
- **Baseline (DeepLabV3)** - *submitted by kostovskaa on April 3nd, 2021 with Avg IoU performance 0.76787 [avg IoU aguadas 0.98478; avg. IoU of buildings 0.66005; avg IoU of platfroms 0.65877]*: Deeplabv3-ResNet101 is constructed by a Deeplabv3  model ([Chen et al. 2017](https://arxiv.org/abs/1706.05587v3){:target="_blank"}) with a ResNet-101 backbone. The pre-trained model has been trained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.  The model has been fine-tuned using only lidar data.


