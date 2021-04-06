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
- July 8th, 2021 - **Official challenge results** - Official results are publised. Prizes are awarded
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

### Public leaderboard

<iframe src="https://competitions.codalab.org/competitions/leaderboard_widget/30429/" style="height: 500px; width: 100%; border: none;"></iframe>


##### Baselines

- **Default Baseline** (submission with empty masks) : submitted by *simidjievskin on April 2nd, 2021 with Avg IoU performance 0.5745 [avg IoU aguadas 0.9634; avg. IoU of buildings 0.4268; avg IoU of platfroms 0.3333]*
- **DeepLabV3 Baseline** - *submitted by kostovskaa on April 3nd, 2021 with Avg IoU performance 0.76787 [avg IoU aguadas 0.98478; avg. IoU of buildings 0.66005; avg IoU of platfroms 0.65877]*: Deeplabv3-ResNet101 is constructed by a Deeplabv3  model ([Chen et al. 2017](https://arxiv.org/abs/1706.05587v3){:target="_blank"}) with a ResNet-101 backbone. The pre-trained model has been trained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.  The model has been fine-tuned using only lidar data.


