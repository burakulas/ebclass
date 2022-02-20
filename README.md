# ebclass
### A Deep Learning Neural Network Algorithm for Classification of Eclipsing Binary Light Curves

We present an image classification algorithm using deep learning convolutional neural network architecture, which classifies the morphologies of eclipsing binary systems based on their light curves. The algorithm trains the machine with light curve images of eclipsing binary stars in three morphological types, contact, detached and semi-detached, whose light curves are provided by three catalogs: Kepler, ASAS and CALEB. The architecture is selected among 132 networks. Our results show that the algorithm estimates the morphological classes of an independent dataset with an accuracy of 91\%.
<br>
###### *Files in main branch*:

**ebclass.py**: Python code applying the classification (edit it according to your dataset location).

**train_set.tar.gz**: Training dataset.

**validation_set.tar.gz**: Validation dataset.
<br>
<br>
###### *Light curve data are from*:

- Kirk, B. et al. 2016. AJ, 151, 68. (http://keplerebs.villanova.edu)

- Pojmanski, G. 1997. AcA 47, 467. (http://www.astrouw.edu.pl/asas/)

- Bradstreet, D. H. et al. 2004. AAS, 204, 501. (http://caleb.eastern.edu)
