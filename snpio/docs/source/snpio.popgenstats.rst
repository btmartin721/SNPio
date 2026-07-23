snpio.popgenstats package
=========================

See :doc:`linkage_disequilibrium` for the unphased LD and recent effective
population-size algorithm, and :doc:`tutorial` for a worked example.

.. automodule:: snpio.popgenstats.pop_gen_statistics

The public LD entry point is
:meth:`snpio.PopGenStatistics.calculate_linkage_disequilibrium`, which returns
:class:`snpio.LinkageDisequilibriumResult`. The lower-level estimator remains
available from ``snpio.popgenstats.linkage_disequilibrium`` for advanced
integration and validation work.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   PopGenStatistics

.. currentmodule:: snpio.popgenstats.linkage_disequilibrium

.. autosummary::
   :toctree: generated/
   :nosignatures:

   LinkageDisequilibriumResult
   LinkageDisequilibrium
