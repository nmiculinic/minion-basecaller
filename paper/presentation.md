---
presentation:
  width: 800
  height: 600
  enableSpeakerNotes: true
---

<!-- slide data-notes:"DNA, the Life&#39;s molecule. For humans it&#39;s 3 billion nucleotides determine who you are. Whether you&#39;re sport superstar, schicofrenic or simply normally maladjusted to modern life. Your eye color, height, and many other features. Its fundamenatal to our functioning and intelligence. If I were to store uncompressed raw human DNA on hard disk it&#39;s only 1 GB, same as Arch linux compressed, yet much more powerful" -->
# MinCall --- MinION end2end deep learning basecaller
**Author: Neven Miculinić**
**Mentor: izv. prof. dr. sc. Mile Šikić**

<!-- slide -->
# Why sequence DNA

* medical diagnosis & research
* Virology -- tracking disease outbreaks
* metagenomics
* ...

<!-- slide data-notes: "Because whole genome it is too enormous, first it is copied multiple times, and than a chemical bomb is dropped, creating many small fragment. Next each some fragment is basecalled, aligned to reference genome or assambled de novo. Whole process has great challanges and uncertainties during basecaller training procedure -- Ground truth uncertainty, you don&apos;t really know what you&apos;re sequencing and hoping the sample isn&apos;t contaminated. Next this small fragments could merge in chemical soup and create chimeric reads. And of course, aligner isn&apos;t perfect and it could map small fragment to wrong reference genome part.&#10;Importat things to mention:&#10;    * ground truth uncertainty&#10;    * Chimeric reads&#10;    * Alignment errors"-->
# Shotgun sequencing
![shotgun_sequencing](https://i.ytimg.com/vi/23iCH3mmifU/maxresdefault.jpg)


<!-- slide -->
# MinION
<img src=http://www.biopsci.com/wp-content/uploads/2014/09/minIONhome_left.png align="middle"  width=800px />

<!-- slide -->
# Motivation
* Small and portable
* Real time sequencing
* Long read length (up to 1Mbp)

<!-- slide data-notes:"Imagine this scenario, you have 4 kinds of people in the world, women, men, both young and old. They are in a line passing through a gate, holding hands. Gate which is 6 people thick. Underneath the gate is a scale, measuring their weight. As they are moving on and off the scale, or simply fiddling in place measurements change. Figure out exact people ordering only from their cumulative weights. That&apos;s our problems! The weights are electrical resistance, and people are nucleotides"-->
# Nanopore
<img src=http://labiotech.eu/wp-content/uploads/2016/07/selective-nanopore-sequencing-minion-nottingham.jpg width=8000/>

<!-- slide -->
<img src=https://imgs.xkcd.com/comics/machine_learning.png align="center"/>

<!-- slide -->
# Deep learning
<img src=http://www.amax.com/blog/wp-content/uploads/2015/12/blog_deeplearning3.jpg alt="deep learning"/>
<!-- slide -->
# Convolutional neural networks (CNN)
![CNN](http://cs231n.github.io/assets/cnn/depthcol.jpeg)
<!-- slide -->
# Residual neural networks
<img src=https://codesachin.files.wordpress.com/2017/02/screen-shot-2017-02-16-at-4-53-01-pm.png width=8000 />
<!-- slide data-notes: "In ML we try minimize negative log likelihood. To get to there let&apos;s start with probability our model outputs given sequence. This picture is from language modeling, but serves illustrative purposes. At each time-step we pick a letter or a blank, and probability of their occurance is their product, similarly to naive Bayes. Total probability for exit sequence is given after summing all possible sequences when removing blank and merging adjacent same character give our target sequence. And that&apos;s the probability we want to maximize."-->
# CTC loss
Connectionist Temporal Classification

<img src=https://raw.githubusercontent.com/baidu-research/warp-ctc/master/doc/deep-speech-ctc-small.png width=800 />

<!-- slide -->

# Final model

* 270 layers divided in 3 block
* Block == 30 residual layers; MaxPool
* gated residual layer == 3x(Relu-BN-Conv1D)
* 3M parameters

<!-- slide -->
# CIGAR
```
Reference: ACGTG_
Read     : A_GTTC
CIGAR    : =D==XI
```

<!-- slide data-notes: "Explain Insertions, deletions, match. Commend on read length"-->

# Results(Mean)
|            | Del | Ins | Match  | Mis | Read length |
|------------|-----------|------------|------------|----------|-------------|
| albacore   | 6%        | 7%         | 86.7%      | 6.3%     | 9843        |
| nanonet    | 8.8%      | 4%         | 89.7%      | 6.2%     | 5029        |
| metrichorn | 8.7%      | 4%         | 89.6%      | 6.3%     | 9262        |
| mincall    | 7.7%      | 4%         | **90.5%**  | 5.6%     | 9378        |

<!-- slide -->

# Results(Median)
|            | Del  | Ins  | Match      | Mis      | Read length |
|------------|------|------|------------|----------|-------------|
| albacore   | 5.6% | 6.8% | 87.1%      | 5.9%     | 6509        |
| nanonet    | 8.4% | 3.6% | 90.2%      | 5.9%     | 3519        |
| metrichorn | 8.3% | 4%   | 90.2%      | 5.9%     | 6138        |
| mincall    | 7.2% | 3.7% | **91.0%**  | 5.2%     | 6224        |

<!-- slide -->
# Results
![kde_match plot](http://i.imgur.com/zaBWYED.png)

<!-- slide data-notes: "Explain what consensus is, and how it's created"-->
# Consensus

| |**average coverage**|**correct**
:-----:|:-----:|:-----:|
metrichorn|10.29|99.78%
nanonet|5.26|99.01%
albacore|10.30|99.84%
mincall|10.30|**99.86%**

<!-- slide -->

# Speed

|            | Type         | Speed (base pairs/second)| E. coli 30x cov|
|------------|--------------|-----------|----|
| albacore*  | 32 core CPU  | 39036     |1h|
| nanonet*   | 32 core CPU  | 4243      |9.1h|
| metrichorn*| Cloud        | unknown   |unknown|
| mincall    | Titan X Black| 3700      |10.4h|



<!-- slide data-notes: "Say thanks to people"-->
# Acknowledgments
* Mentor izv. prof. dr. sc. Mile Šikić
* colleague Marko Ratković
* Other helping people: Fran Jurišić, Ana Marija Selak, Ivan Sović and Martin Šošić.
