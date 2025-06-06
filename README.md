# ebclass
### A Deep Learning Neural Network Algorithm for Classification of Eclipsing Binary Light Curves

**ebclass** is an image classification algorithm with deep learning convolutional neural network architecture classifiying the eclipsing binary systems morphologically based on their light curves. The algorithm classifies the light curve images of eclipsing binary stars in three morphological types, contact, detached and semi-detached. Results show that it estimates the morphologies of an independent dataset with an accuracy of 92\%.
<br>
###### *Files in main branch*:

**ebclass.py**: Python code applying the classification.

**model.tar.gz**: .hd5 model file.

**training_set.tar.gz**: Training dataset. (splitted into 2 parts, to join: `cat training_set.a* > training_set.tar.gz`)

**validation_set.tar.gz**: Validation dataset.
<br>
<br>
###### *Light curve data are from*:

- Kirk, B. et al. 2016. AJ, 151, 68. [(Kepler EBC)](http://keplerebs.villanova.edu)

- Pojmanski, G. 1997. AcA 47, 467. [(ASAS)](http://www.astrouw.edu.pl/asas/)

- Bradstreet, D. H. et al. 2004. AAS, 204, 501. [(CALEB](http://caleb.eastern.edu), former [EBOLA)](https://ui.adsabs.harvard.edu/abs/2004AAS...204.0501B/abstract)

