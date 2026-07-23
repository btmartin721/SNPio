Publication, Citation, and Reproducibility
==========================================

How to cite SNPio
-----------------

Please cite the published SNPio article when using the software:

  Martin, B.T., Monaco, D.R., Sharabi, N., Mussmann, S.M., and Chafin, T.K.
  SNPio: a Python interface for population genomic data processing.
  *BMC Bioinformatics* (2026).
  `https://doi.org/10.1186/s12859-026-06546-5
  <https://doi.org/10.1186/s12859-026-06546-5>`_.

BibTeX
^^^^^^

.. code-block:: bibtex

   @article{MartinEtAl2026,
     author  = {Martin, Bradley T. and Monaco, Domenico R. and
                Sharabi, Nadine and Mussmann, Steven M. and
                Chafin, Tyler K.},
     title   = {{SNPio}: a Python interface for population genomic data
                processing},
     journal = {BMC Bioinformatics},
     year    = {2026},
     doi     = {10.1186/s12859-026-06546-5},
     url     = {https://doi.org/10.1186/s12859-026-06546-5}
   }

When reporting unphased LD or LD-based recent :math:`N_e`, also cite the
estimator methodology :cite:p:`RagsdaleGravel2020`.

Publication files
-----------------

The DOI above is the authoritative Version of Record. The accepted,
not-yet-typeset manuscript supplied with this project remains available for
long-term reference:

`SNPio: a Python interface for population genomic data processing
<./_static/papers/SNPioManuscript_NotTypesetYet.pdf>`_.

The repository copy is covered by the checksum in
``_static/papers/SHA256SUMS``. It is a documentation artifact and is not read
by SNPio at runtime.

OSF reproducibility repository
------------------------------

Data, analysis materials, and reproducibility resources associated with SNPio
are archived in the Open Science Framework repository:

`https://doi.org/10.17605/OSF.IO/WDQ3F
<https://doi.org/10.17605/OSF.IO/WDQ3F>`_.

The repository's executable feature-validation code is maintained under
`snpio/validation/
<https://github.com/btmartin721/SNPio/tree/master/snpio/validation>`_. Curated,
checksummed validation evidence intended for users and reviewers is maintained
under the root `validation/
<https://github.com/btmartin721/SNPio/tree/master/validation>`_ directory.

Installation distributions
--------------------------

SNPio is distributed through all of the following maintained channels:

PyPI
   `SNPio on PyPI <https://pypi.org/project/snpio/>`_

   .. code-block:: shell

      python -m pip install snpio

conda
   `SNPio on Anaconda
   <https://anaconda.org/channels/btmartin721/packages/snpio/overview>`_

   .. code-block:: shell

      conda create -n snpio-env python=3.12
      conda activate snpio-env
      conda install -c btmartin721 snpio

Docker
   `SNPio on DockerHub <https://hub.docker.com/r/btmartin721/snpio>`_

   .. code-block:: shell

      docker pull btmartin721/snpio:latest
      docker run -it btmartin721/snpio:latest

Use a virtual environment or conda environment for native installations. On
Windows, use WSL or Docker for the Unix-oriented workflow.
