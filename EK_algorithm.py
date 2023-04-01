import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import csv
import networkx as nx

""" We use the network x package to run the Edmonds-Karp algorithm.
    This is available here:
    https://networkx.org/documentation/stable/reference/introduction.html
"""
np.set_printoptions(suppress=True)

def open_csv_col(file_name):
    """ A function to open the csv file and return it as a matrix
    
        Inputs:
            file_name:  the name of the file to segment
        
        Returns:
            img_data:   the image data as a matrix
    """
    
    f = open('{}_data_col.csv'.format(file_name))
    csv_reader = csv.reader(f)
    
    img_data = []
    for line in csv_reader:
        new_line = []
        for element in line:
            pixel = element[1:-1].split()
            pixel = [eval(i) for i in pixel]
            new_line.append(pixel)
    
        img_data.append(new_line)
        
    return img_data

def pairwise_capacity(Ii, Ij, sigma): 
    """ A function to define the pariwsie capacity of an edge (i,j)
    
        Inputs:
            Ii, Ij:     the colour of the two nodes i and j
            sigma:      the parameter sigma
        
        Returns:
            B:          the pariwise capacity of the edge
    """
        
    col_diff = (Ii[0] - Ij[0])**2 + (Ii[1] - Ij[1])**2 + (Ii[2] - Ij[2])**2
    B = np.exp(-col_diff/(6*sigma**2))
    return B

def create_graph(data, alpha, sigma, seeds):
    """ This function creates a graph for the image. We assume here a square 
        image, although this is easily generalised to non-square images.
        
        Inputs:
            data:   the image data
            alpha:  parameter alpha
            sigma:  parameter sigma
            seeds:  the regions of the image that we takes as the foreground
                and background seeds
        
        Returns:
            G:      the graph that represents the image
    """
    
    G = nx.DiGraph()
    n = len(data)
    m = n-1 #index
    print()
  
    """Implementation of user seeds"""
    
    """White Seed"""
    seeds = [int(np.round(seeds[i] * n/100)) for i in range(len(seeds))]
    n_w, m_w, xpos_w, ypos_w, n_b, m_b, xpos_b, ypos_b = seeds
    
    seed_w = []
    histogram_w = []
    for i in range(n_w):
        for j in range(m_w):
            seed_w.append("{},{}".format(ypos_w + i, xpos_w + j))
            histogram_w.append(data[ypos_w + i][xpos_w + j])
            
    mean_w = np.mean(histogram_w, axis = 0)
    sd_w = np.std(histogram_w, axis = 0)
    
    """Black Seed"""
    seed_b = []
    histogram_b = []
    for i in range(n_b):
        for j in range(m_b):
            seed_b.append("{},{}".format(ypos_b + i, xpos_b + j))
            histogram_b.append(data[ypos_b + i][xpos_b + j])
            
    mean_b = np.mean(histogram_b, axis = 0)
    sd_b = np.std(histogram_b, axis = 0)
    
    """Pairwise links"""
    for i in range(m):
        for j in range(m):
            G.add_edge("{},{}".format(i,j), "{},{}".format(i+1,j), capacity = pairwise_capacity(data[i][j], data[i+1][j], sigma))
            G.add_edge("{},{}".format(i,j), "{},{}".format(i,j+1), capacity = pairwise_capacity(data[i][j], data[i][j+1], sigma))

    """Unary links"""
    for i in range(n):
        for j in range(n):
            diff_s = [norm.pdf(data[i][j][k], mean_w[k], sd_w[k]) for k in range(3)]
            diff_t = [norm.pdf(data[i][j][k], mean_b[k], sd_b[k]) for k in range(3)]
            G.add_edge("s", "{},{}".format(i,j), capacity = -alpha*np.log(sum(diff_s)/3))
            G.add_edge("{},{}".format(i,j), "t" , capacity = -alpha*np.log(sum(diff_t)/3))
            
    """Seed blocks"""
    max_cap = max([G[edge[0]][edge[1]]["capacity"] for edge in G.edges])
    K = 1 + max_cap 
    for node in seed_w:
        G["s"][node]['capacity'] = 0
        G[node]["t"]['capacity'] = K

    for node in seed_b:
        G["s"][node]['capacity'] = K
        G[node]["t"]['capacity'] = 0

    return G

def plot_image(data, partition):
    """ Plots the original and segmented image.
    
        Inputs:
            data:       the original image
            partition:  a tuple containing a list of both the foreground
                        and background parts of the image
        """
    
    n = len(data)
    col_flag = 0 # we initally colour the foreground (black) parts of the image
    data_sorted = np.zeros((n,n))
    for side in partition:
        for i in side:
            if i == "s" or i == "t": # we do not plot the source or the sink
                pass
            
            else:
                loc = [int(i.split(",")[0]), int(i.split(",")[1])]
                data_sorted[loc[0]][loc[1]] = col_flag
        
        col_flag = 1 # next we colour the background (white)
    
    pixel_plot = plt.figure()
    pixel_plot = plt.imshow(
      data, cmap = "binary", interpolation='nearest')
    plt.axis('off')
    plt.show(pixel_plot)
    
    pixel_plot = plt.figure()
    pixel_plot = plt.imshow(
      1 - data_sorted, cmap = "binary", interpolation='nearest')
    plt.axis('off')
    plt.show(pixel_plot)
    
