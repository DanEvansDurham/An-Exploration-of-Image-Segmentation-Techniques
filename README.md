# Level-3-Project
Code repository for the image segmentation algorithms used in my level 3 project report.

The file run_algorithms runs both of the algorithms on the provided test image. 

The files conv_img and conv_img_col convert the test image into a csv file and reduce it by the scaling factor. conv_img converts to greyscale and conv_img_col keeps the full colour image. 

The file EK_algorithm runs the Edmonds-Karp algorithm. To do this, we use the minimum cut function avaialbale from the networkx package:
    https://networkx.org/documentation/stable/reference/introduction.html

The file MBMF_algorithm runs the MBMF algorithm, using code which is a lightly edited version of the one by Yuan et al. which is available here: 
https://uk.mathworks.com/matlabcentral/fileexchange/34126-fast-continuous-max-flow-algorithm-to-2d-3d-image-segmentation

based on their paper:
Jing Yuan, Egil Bae, and Xue-Cheng Tai. A study on continuous max-
ow and
min-cut approaches. In Conference on Computer Vision and Pattern Recognition,
2010.

Direct any questions to daniel.c.evans@durham.ac.uk
