# Optimal Sketching for Trace Estimation

### Requirements
* Python 3.6+
* ``pip install -r requirements.txt``
* Install ``sparse_dot_mkl``: a Python wrapper for Intel Math Kernel Library (MKL) that supports fast sparse matrix multiplication.<br/>
Please check out: [sparse-dot-mkl 0.7.1](https://pypi.org/project/sparse-dot-mkl/)

### Datasets
All datasets used in the experiments are in the ``data`` folder. <br />
* ``roget`` (Roget’s Thesaurus semantic graph, [Link](http://vlado.fmf.uni-lj.si/pub/networks/data/)): <br />
    ``data/Roget.net`` is the raw graph data. <br />
    ``data/roget.py`` parses the raw data and generates the connectivity graph. <br />
* ``arxiv_cm`` (arXiv Condensed Matter collaboration network, [Link]( https://snap.stanford.edu/data/ca-CondMat.html)): <br />
    ``data/ca-CondMat.txt`` is the raw graph data. <br />
    ``data/arxiv_wiki.py`` parses the raw data and generates the connectivity graph. <br />
* ``precipitation`` (2010 US day precipitation data, [Link](https://catalog.data.gov/dataset/u-s-hourly-precipitation-data)): <br />
    ``process_precipitation.ipynb`` is the script for constructing a covariance matrix based on 1000 subsampled data points using RBF kernel with length scale 1, from a processed version of this dataset ``processed-data-2010-jan.csv``. <br />
    The processed version, ``processed-data-2010-jan.csv``, can be found in the GitHub Repo [GPML_SLD](https://github.com/kd383/GPML_SLD) of paper [Scalable Log Determinants for Gaussian Process Kernel Learning](https://arxiv.org/abs/1711.03481).
* ``bcsstk20.mat`` and ``bcsstm08.mat`` are two matrices from the UFL Sparse Matrix Collection ([Link](https://sparse.tamu.edu/)) used to compare performances on log determinant estimation.

### Algorithms
The ``algorithms`` folder contains the implementation of the three trace estimation algorithms: ``NA-Hutch++``, ``Hutch++`` and ``Hutchinson``. <br />
* ``algos.py``: A primitive implementation of the three trace estimation algorithms. Each algorithm has a direct access to the input matrix. Note this version is not used directly in the experiments.
* ``algos_oracle.py``: The sequential implementation of the three trace estimation algorithms. Each algorithm has access to matrix-vector oracles.
* ``algos_parallel.py``: The parallel implemntation of the three trace estimation algorithms via ``multiprocessing``. Each algorithm has access to matrix-vector oracles.

Algorithms for comparing performance of log determinant estimation:
* ``logdet/maxent_logdet_seq.py`` and ``stable_maxent.py`` implements the maximum entropy estimation based 
log determinant estimation from [Entropic Trace Estimates for Log Determinants](https://arxiv.org/abs/1704.07223).
The code is a direct translation of the original Matlab code from 
[https://github.com/OxfordML/EntropicTraceEstimation](https://github.com/OxfordML/EntropicTraceEstimation)

### Experiment Scripts
* ``exp_synthetic.py``: Conducting experiments on the synthetic data. The synthetic data will be generated when performing the experiment.
* ``exp_large_arxiv.py``: Conducting experiments on ``arxiv_cm``.
* ``exp_roget.py``: Conducting experiments on ``roget``.
* ``exp_precipitation.py``: Conducting experiments on ``precipitation``.
* ``logdet/check_logdet.py``: Conducting experiments to compare trace estimation algorithms as subroutines to estimate 
the moments of eigenvalues used for log determinant estimation. 

### Utilities
* ``lanczos_np.py``: The Lanczos algorithm subroutine that is used as a matrix-vector multiplication oracle. 
*  ``utils.py``: All experiment supports, including timing and comparing parallel and sequential algorithms.
* ``plot_results.py``: Generate resulting plots.
* ``logdet/data_utils.py``: Read sparse input matrix.
* ``logdet/eigenspectrum.py``: Extract the eigenvalues of the input matrix.
* ``logdet/plot_logdet_results.py``: Generate resulting plots on log determinant estimation.

### Results
All experiment results are included in the ``results`` and ``logdet/results`` folder.

### Plots
All resulting plots are included in the ``plots`` and ``logdet/plots`` folder.


