DIBCO Evaluation Tool

<4 Inputs> that correspond to filenames of:
I.1 GT image
I.2 Binarized image for evaluation
I.3 "Recall Weights"** .dat file
I.4 "Precision Weights"** .dat file

<8 Outputs>:
O.1 F-Measure
O.2 pseudo F-Measure (Fps)**
O.3 PSNR
O.4 DRD***
O.5 Recall
O.6 Precision
O.7 pseudo-Recall (Rps)**
O.8 pseudo-Precision (Pps)**

*Notice that for inputs I.3, I.4, the program 'BinEvalWeights.exe' should be used to generate the .dat files containing the "Recall/Precision weights".

Example Run:
>DIBCO_metrics PR_GT.tiff PR_bin.bmp PR_RWeights.dat PR_PWeights.dat

F-Measure               :       93.5987
pseudo F-Measure (Fps)  :       97.6863
PSNR                    :       15.8163
DRD                     :       1.8681
Recall                  :       90.4607
Precision               :       96.9623
pseudo-Recall (Rps)     :       99.4464
pseudo-Precision (Pps)  :       95.9875


*ATTENTION: When providing the .dat files, always provide first the "Recall weights" file (PR_RWeights.dat)
and thereafter the "Precision weights" file (PR_PWeights.dat).

<RELATED PUBLICATIONS>

**K. Ntirogiannis, B. Gatos and I. Pratikakis,
"Performance Evaluation Methodology for Historical Document Image Binarization",
IEEE Trans. Image Proc., vol.22, no.2, pp. 595-609, Feb. 2013.

***H.Lu, A.C. Kot and Y.Q. Shi,
"Distance Reciprocal Distortion Measure for Binary Document Images",
IEEE Sigal Proc. Lett., vol.11, no.2, pp. 228-231, Feb. 2004.