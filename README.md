# DANN-Based-DA-for_CD-NOISY-LABELS
The current project conatins the code to perform Domain Adaptation based on Domain-Adversarial Neural Network (DANN) [1] for change detection in remote sensing images, specifically for deforestation detection in tropical forests such as the Amazon rainforest and the Brazilian savannah.

The following figure, inspired by [1], shows the proposed methodology. The domain adaptation process begins by selecting class-wise balanced training samples from both domains. A traditional down/up sampling strategy can be adopted for the source domain because the class labels are available. However, such balancing procedure can not be applied straightforwardly for the target domain due the target labels are unknown during training. To overcome that problem, we included a pseudo-labelling scheme based on Change Vector Analysis (CVA) and a thresholding technique based on the OTSU to produce a pseudo label map used to select a less imbalanced training set in target domains (a).

Then, the DANN model is trained with selected samples until convergence, as illustrated in (b). Once the model’s functions have been trained, Gf and Gc are used to classify the samples extracted from the target testing images (c).

![Image](DANN_CD.JPG)

# Prerequisites
1- Python 3.7.4

2- Tensorflow 1.14

# Dataset
Such implementation has been evaluated in a change detection task namely deforestation detection and aiming at reproducing the results obtained in [2] and [3] we make available the images used in this project which can be found in the following links for the [Amazon Biome](https://drive.google.com/drive/folders/1V4UdYors3m3eXaAHXgzPc99esjQOc3mq?usp=sharing) as well as for the [Cerrado](https://drive.google.com/drive/folders/14Jsw0LRcwifwBSPgFm1bZeDBQvewI8NC?usp=sharing). In the same way, the references can be obtained by clicking in [Amazon references](https://drive.google.com/drive/folders/15i04inGjme56t05gk98lXErSRgRnU30x?usp=sharing) and [Cerrado references](https://drive.google.com/drive/folders/1n9QZA_0V0Xh8SrW2rsFMvpjonLNQPJ96?usp=sharing).

# References

[1] Ganin and V. Lempitsky,“Unsupervised   domain   adaptation  by backpropagation,”arXiv preprint arXiv:1409.7495, 2014.

[2] Vega, P. J. S. (2021). DEEP LEARNING-BASED DOMAIN ADAPTATION FOR CHANGE DETECTION IN TROPICAL FORESTS (Doctoral dissertation, PUC-Rio).
