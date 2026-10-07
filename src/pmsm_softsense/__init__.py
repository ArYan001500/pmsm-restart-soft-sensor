"""Restart (re-initialisation) of rotor-temperature soft sensors after estimator resets.

Modules
-------
realdata     loader of the Paderborn PMSM dataset with a test-lock firewall
lptn_real    four-node rate-form lumped-parameter thermal network (LPTN)
lptn_fit     LPTN identification: equation-error NNLS and output-error refinement
reset_tools  LPTN simulator, Kalman filter and simulated operating histories (CBP training data)
"""

__version__ = "1.0.0"
