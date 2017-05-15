<!-- slide -->
# MinCall --- MinION end2end deep learning basecaller

**Mentor: izv. prof. dr. sc. Mile Sikić**
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

Speaking: desribing metodology, nanopore and how it functions
<!-- slide -->
# Motivation
* small and portable
* Real time sequencing
* Long read length (up to 1Mbp)
<!-- slide -->
![](https://i.imgflip.com/1p0y57.jpg)
<!-- slide -->
# Deep learning
![](http://www.amax.com/blog/wp-content/uploads/2015/12/blog_deeplearning3.jpg)
<!-- slide -->
# Convolutional neural networks (CNN)
![](http://cs231n.github.io/assets/cnn/depthcol.jpeg)
<!-- slide -->
# Residual neural networks
![](https://codesachin.files.wordpress.com/2017/02/screen-shot-2017-02-16-at-4-53-01-pm.png)
<!-- slide -->
# Connectionist Temporal Classification (CTC) loss
![](https://raw.githubusercontent.com/baidu-research/warp-ctc/master/doc/deep-speech-ctc-small.png)

<!-- slide -->
# Results
|          | Deletions | Error | Insertions | Match  | Mismatch | Read length |
|----------|-----------|-------|------------|------------|----------|-------------|
| albacore | 6%        | 19.4% | 7%         | 86.7%      | 6.3%     | 9843        |
| nanonet  | 8.8%      | 19%   | 4%         | 89.7%      | 6.2%     | 5029        |
| mincall  | 7.7%      | 17.2% | 4%         | 90.5%      | 5.6%     | 9378        |
| metrichorn  | 8.7%      | 19.1% | 4%         | 89.6%      | 6.3%     | 9262        |
<!-- slide -->
# Results
![](http://i.imgur.com/zaBWYED.png)
<!-- slide -->
# Acknowledgments
* Mentor izv. prof. dr. sc. Mile Sikić
* colleague Marko Ratkovic
* Other helping people: Fran Jurišić, Ana Marija Selak, Ivan Sović and Martin Šošić.
