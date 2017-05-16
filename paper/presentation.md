---
presentation:
  width: 800
  height: 600
  enableSpeakerNotes: true
---

<!-- slide data-notes:"DNA, the Life&#39;s molecule. For humans it&#39;s 3 billion nucleotides determine who you are. Whether you&#39;re sport superstar, schicofrenic or simply normally maladjusted to modern life. Your eye color, height, and many other features. Its fundamenatal to our functioning and intelligence. If I were to store uncompressed raw human DNA on hard disk it&#39;s only 1 GB, same as Arch linux compressed, yet much more powerful" -->
# MinCall --- MinION end2end deep learning basecaller

**Mentor: izv. prof. dr. sc. Mile Sikić**

<!-- slide -->
# Why sequence DNA

* medical diagnosis & research
* Virology -- tracking disease outbreaks
* metagenomics
* ...

<!-- slide data-notes: "Because whole genome it is too enormous, first it is copied multiple times, and than a chemical bomb is dropped, creating many small fragment. Next each some fragment is basecalled, aligned to reference genome or assambled de novo. Whole process has great challanges and uncertainties during basecaller training procedure -- Ground truth uncertainty, you don't really know what you're sequencing and hoping the sample isn't contaminated. Next this small fragments could merge in chemical soup and create chimeric reads. And of course, aligner isn't perfect and it could map small fragment to wrong reference genome part."-->
# Shotgun sequencing
![shotgun_sequencing](https://i.ytimg.com/vi/23iCH3mmifU/maxresdefault.jpg)

<!-- slide -->
# Challenges
* ground truth uncertainty
* Chimeric reads
* Alignment errors

<!-- slide -->
# MinION
![](http://www.biopsci.com/wp-content/uploads/2014/09/minIONhome_left.png)

<!-- slide -->
# Motivation
* Small and portable
* Real time sequencing
* Long read length (up to 1Mbp)

<!-- slide -->
# Nanopore
![nanopore](http://labiotech.eu/wp-content/uploads/2016/07/selective-nanopore-sequencing-minion-nottingham.jpg)

<!-- slide -->
![brace yourself math is comming](https://i.imgflip.com/1p0y57.jpg)

<!-- slide -->
# Deep learning
![deep learning overview](http://www.amax.com/blog/wp-content/uploads/2015/12/blog_deeplearning3.jpg)
<!-- slide -->
# Convolutional neural networks (CNN)
![CNN](http://cs231n.github.io/assets/cnn/depthcol.jpeg)
<!-- slide -->
# Residual neural networks
![ResNet](https://codesachin.files.wordpress.com/2017/02/screen-shot-2017-02-16-at-4-53-01-pm.png)
<!-- slide -->
# Connectionist Temporal Classification (CTC) loss
![CTC loss](https://raw.githubusercontent.com/baidu-research/warp-ctc/master/doc/deep-speech-ctc-small.png)

<!-- slide data-notes: "Explain Insertions, deletions, match. Commend on read length"-->
# Results
|            | Deletions | Insertions | Match  | Mismatch | Read length |
|------------|-----------|------------|------------|----------|-------------|
| albacore   | 6%        | 7%         | 86.7%      | 6.3%     | 9843        |
| nanonet    | 8.8%      | 4%         | 89.7%      | 6.2%     | 5029        |
| metrichorn | 8.7%      | 4%         | 89.6%      | 6.3%     | 9262        |
| mincall    | 7.7%      | 4%         | **90.5%**  | 5.6%     | 9378        |

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

<!-- slide data-notes: "Say thanks to people"-->
# Acknowledgments
* Mentor izv. prof. dr. sc. Mile Sikić
* colleague Marko Ratković
* Other helping people: Fran Jurišić, Ana Marija Selak, Ivan Sović and Martin Šošić.
