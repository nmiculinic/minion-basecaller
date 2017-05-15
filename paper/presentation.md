<!-- slide -->
# MinCall --- MinION end2end deep learning basecaller

**Mentor: izv. prof. dr. sc. Mile Sikić**

*speaking text, to be removed:* DNA, the Life's molecule. It's 3 billion nucleotides determine who you are. Whether you're phychopath, schicofrenic or simply normally maladjusted to modern life. Your eye color, height, and many other features. It's fundamenatal to our functioning and intelligence. If I were to store uncompressed raw human DNA on hard disk it's only 1 GB, same as Arch linux compressed
<!-- slide -->
# Why sequence DNA

* medical diagnosis
* Virology -- tracking disease outbreaks
* metagenomics
* Learning how we function
* ...

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
* small and portable
* Real time sequencing
* Long read length (up to 1Mbp)

<!-- slide  -->
# Shotgun sequencing
![shotgun_sequencing](https://i.ytimg.com/vi/23iCH3mmifU/maxresdefault.jpg)

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

<!-- slide -->
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

<!-- slide -->
# Results

| |**average coverage**|**insertion**|**deletion**|**correct**
:-----:|:-----:|:-----:|:-----:|:-----:
metrichorn|10.29|2%|20%|99.78%
nanonet|5.26|15%|75%|99.01%
albacore|10.30|2%|9%|99.84%
mincall|10.30|1%|13%|**99.86%**

<!-- slide -->
# Acknowledgments
* Mentor izv. prof. dr. sc. Mile Sikić
* colleague Marko Ratković
* Other helping people: Fran Jurišić, Ana Marija Selak, Ivan Sović and Martin Šošić.
