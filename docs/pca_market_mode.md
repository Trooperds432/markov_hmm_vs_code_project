# PCA Market Mode

PCA market-mode features estimate whether the asset is being dominated by a common market factor.
The implementation uses a trailing window of asset, benchmark, and available proxy returns. If fewer
than three valid series exist, the block writes structured warnings and skips PCA values.

`pc1_explained_variance` near or above `0.55` means the first component explains a large share of
recent variance. In that setting, a single-stock HMM signal may be mostly market beta rather than
idiosyncratic regime information.
