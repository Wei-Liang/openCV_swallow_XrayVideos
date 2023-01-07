Data: raw X-rays videos of swallowing from different subjects including patients and non-patients

First goal:
Comparing between several preprocessing methods to enhance X-ray video contrast and facilitate comparison among subjects:
(1) Naive Histogram Equalizer, which is a simple stretch of the range of brightness
(2) variants of Contrast Limited Adaptive Histogram Equalization (CLAHE), with different clip limits and stride sizes



Second goal:
Tracking certain orofacial structures (eg. chin and spine), from a query image.
Scale-Invariant Feature Transform (SIFT) is used to to extract keypoints that are scale-invariant. Then I use knn to match feature points. If enough matching points are found (10 here), they can be passed to find the perspective transformation (which is a 3X3 matrix).  
The query image (with persepctive transformation applied) is considered to have been found in the frame. Locations of all the pairs of matching points are connected with lines to show match. 


Methods based on OpenCV