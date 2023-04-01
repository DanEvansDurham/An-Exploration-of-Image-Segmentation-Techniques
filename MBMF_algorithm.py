import numpy as np
import csv
import matplotlib.pyplot as plt

def CMF_2D(ur, params = [500, 1e-4, 0.3, 0.25], λlab = [0,1], α=None, Cs=None, Ct=None):
    """ Function to run the MBMF algorithm. This code was taken from the 
        paper by Yuan et al. and is available here:
        https://www.mathworks.com/matlabcentral/fileexchange/34126-fast-continuous-max-flow-algorithm-to-2d-3d-image-segmentation
    
        Inputs:
            ur:         the matrix of image data
            params:     the parameters for the algorithm
            λlab:       the value of the λlab parameter
            α:          the value of the α parameter
            Cs, Ct:     option to set Cs and Ct if desired
        
        Returns:
            λ:          matrix of values of λ
            erriter:    the error level of the final iteration 
                        (indicates convergence)
            numIter:    the number of iterations run
    """
    
    print()
    print("λlab value: {}, alpha: {}".format(λlab, α))
    print()
    numIter, errbound, cc, steps = params
    rows, cols = ur.shape
    imgSize = ur.size
    assert(rows*cols == imgSize)

    if α is None:
        α = 0.5 * np.ones((rows, cols))

    # Build the data terms
    if Cs is None:
        Cs = np.abs(ur - λlab[0])
    if Ct is None:
        Ct = np.abs(ur - λlab[1])

    # set the initial values
    # - the initial value of λ is set to be an initial cut, see below.
    # - the initial values of two terminal flows ps and pt are set to be the
    # specified legal flows.
    # - the initial value of the spatial flow fields p = (pp1, pp2) is set to
    # be zero.
    λ = np.asarray((Cs - Ct) >= 0, np.float64)
    ps = np.minimum(Cs, Ct)  # Minimum element wise between the two
    pt = ps

    pp1 = np.zeros((rows, cols+1))
    pp2 = np.zeros((rows+1, cols))
    divp = np.zeros((rows, cols))

    erriter = np.zeros(numIter)

    for i in range(numIter):
        # update the spatial flow field p = (pp1, pp2):
        # the following steps are the gradient descent step with steps as the
        # step-size.
        pts = divp - (ps - pt + λ/cc)
        pp1[:, 1:-1] += steps * (pts[:, 1:] - pts[:, :-1])
        pp2[1:-1, :] += steps * (pts[1:, :] - pts[:-1, :])

        # the following steps give the projection to make |p(x)| <= α(x)
        squares = pp1[:, :-1]**2 + pp1[:, 1:]**2 + pp2[:-1, :]**2 + pp2[1:, :]**2
        gk = np.sqrt(squares * .5)
        gk = (gk <= α) + np.logical_not(gk <= α) * (gk / α)
        gk = 1 / gk

        pp1[:, 1:-1] = (.5 * (gk[:, 1:] + gk[:, :-1])) * (pp1[:, 1:-1])
        pp2[1:-1, :] = (.5 * (gk[1:, :] + gk[:-1, :])) * (pp2[1:-1, :])

        divp = pp1[:, 1:] - pp1[:, :-1] + pp2[1:, :] - pp2[:-1, :]

        # updata the source flow ps
        pts = divp + pt - λ/cc + 1/cc
        ps = np.minimum(pts, Cs)

        # update the sink flow pt
        pts = - divp + ps + λ/cc
        pt = np.minimum(pts, Ct)

        errλ = cc * (divp + pt - ps)
        λ -= errλ
            
        erriter[i] = np.sum(np.abs(errλ)) / imgSize

        if erriter[i] < errbound:
            return λ, erriter, i
        
    return λ, erriter, numIter


def open_csv(file_name):
    """ A function to open the csv file and return it as a matrix
    
        Inputs:
            file_name:  the name of the file to segment
        
        Returns:
            img_data:   the image data as a matrix
    """
    
    f = open('{}_data.csv'.format(file_name))
    csv_reader = csv.reader(f)
    
    img_data = []
    for line in csv_reader:
        img_data.append([int(i) for i in line])
        
    return img_data


def plot_2d(λ, data, l):
    """ A function to plot the original image and the segmented image
        
        Inputs:
            λ:      the matrix of λ values
            data:   the original image data
            l:      the threshold parameter l
    """
    
    shape = λ.shape
    segmented = np.zeros((shape[0], shape[1]))

    for i in range(shape[0]):
        for j in range(shape[1]):
            if λ[i][j] < l:
                segmented[i][j] = 0
            
            else:
                segmented[i][j] = 1

    """ We now plot the original image and the segmented image.
        Note that the imshow() package works inversely so we have 0 as 
        white and either 1 or 255 as black. """
        
    pixel_plot = plt.figure()
    pixel_plot = plt.imshow(
      255 - data, cmap = "binary", interpolation='nearest')
    plt.axis('off')
    plt.show(pixel_plot)


    pixel_plot = plt.figure()
    pixel_plot = plt.imshow(
      1 - segmented, cmap = "binary", interpolation='nearest')
    plt.axis('off')
    plt.show(pixel_plot)
    

