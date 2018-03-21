# `cil_project`

## Implementation Details
The current required libraries are the following:

- `numpy`
- `matplotlib`
- `tensorflow` or `theano` or `keras` (Francesco)
- Pillow for `PIL`

### Project Structure
```
- assets
  | - test
  |   | all data required for testing
- papers
  | all relevant papers for this project
- src
  | test.py         -> contains all unit tests
  | utility.py      -> contains all utility functions
- README.md         -> this file
```


## Resources
[This](https://github.com/dariopavllo/road-segmentation) is an example of exactly what we are supposed to do. The technique used can be read in the report within the git repo.

[This paper](http://www.mdpi.com/2072-4292/8/4/329/html) does more broad per-pixel segmentation by using satellite images to classify complete regions. However, this is also applicable for road segmentation.

## Main current ideas:

CNN + 4 filters (computation allowing):

- vertical lines (https://www.coursera.org/learn/convolutional-neural-networks/lecture/4Trod/edge-detection-example) -> learn vertical features as streets
- horizontal lines (same above link) -> learn horizontal features as streets
- diagonal lines filter (same above link) -> learn diagonal features as streets
- color (third dimension) -> learn a color threshold given the background possibly (Bayesian estimation)


Objective: learn the features as independent ones.
->  threshold on top of the layers mixing all channels.


## Papers
The BibTex references bellow refer to papers in `papers/` directory.

- ```
  @article{panboonyuen2017road,
    title={Road Segmentation of Remotely-Sensed Images Using Deep Convolutional Neural Networks with Landscape Metrics and Conditional Random Fields},
    author={Panboonyuen, Teerapong and Jitkajornwanich, Kulsawasd and Lawawirojwong, Siam and Srestasathiern, Panu and Vateekul, Peerapon},
    journal={Remote Sensing},
    volume={9},
    number={7},
    pages={680},
    year={2017},
    publisher={Multidisciplinary Digital Publishing Institute}
  }
  ```
- ```
  @article{bhattacharya1997improved,
    title={An improved backpropagation neural network for detection of road-like features in satellite imagery},
    author={Bhattacharya, U and Parui, SK},
    journal={International Journal of Remote Sensing},
    volume={18},
    number={16},
    pages={3379--3394},
    year={1997},
    publisher={Taylor \& Francis}
  }
  ```
- ```
  @article{mokhtarzade2008automatic,
    title={Automatic road extraction from high resolution satellite images using neural networks, texture analysis, fuzzy clustering   and genetic algorithms},
    author={Mokhtarzade, M and Zoej, MJ Valadan and Ebadi, H},
    journal={The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
    volume={37},
    pages={549--556},
    year={2008}
  }
  ```
- ```
  @article{song2004road,
    title={Road extraction using SVM and image segmentation},
    author={Song, Mingjun and Civco, Daniel},
    journal={Photogrammetric Engineering \& Remote Sensing},
    volume={70},
    number={12},
    pages={1365--1371},
    year={2004},
    publisher={American Society for Photogrammetry and Remote Sensing}
  }
  ```
- ```
  @article{kingma2014adam,
    title={Adam: A method for stochastic optimization},
    author={Kingma, Diederik P and Ba, Jimmy},
    journal={arXiv preprint arXiv:1412.6980},
    year={2014}
  }
  ```
- ```
  @article{xu2015empirical,
    title={Empirical evaluation of rectified activations in convolutional network},
    author={Xu, Bing and Wang, Naiyan and Chen, Tianqi and Li, Mu},
    journal={arXiv preprint arXiv:1505.00853},
    year={2015}
  }
  ```
- ```
  @article{bittel2015pixel,
    title={Pixel-wise segmentation of street with neural networks},
    author={Bittel, Sebastian and Kaiser, Vitali and Teichmann, Marvin and Thoma, Martin},
    journal={arXiv preprint arXiv:1511.00513},
    year={2015}
  }
  ```
- ```
  @article{zhang2006benefit,
    title={Benefit of the angular texture signature for the separation of parking lots and roads on high resolution multi-spectral imagery},
    author={Zhang, Qiaoping and Couloigner, Isabelle},
    journal={Pattern recognition letters},
    volume={27},
    number={9},
    pages={937--946},
    year={2006},
    publisher={Elsevier}
  }
  ```
- ```
  @article{mena2005automatic,
    title={An automatic method for road extraction in rural and semi-urban areas starting from high resolution satellite imagery},
    author={Mena, Juan B and Malpica, Jos{\'e} A},
    journal={Pattern recognition letters},
    volume={26},
    number={9},
    pages={1201--1220},
    year={2005},
    publisher={Elsevier}
  }
  ```

