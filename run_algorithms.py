from Conv_img_col import img_data_col
from Conv_img import img_data
from EK_algorithm import open_csv_col, create_graph, plot_image
from MBMF_algorithm import open_csv, CMF_2D, plot_2d
import networkx as nx
import numpy as np
from time import time

def discrete_segment(data):
    """Discrete Segmentation: Edmonds-Karp Algorithm
    
    Inputs:
        data: the image data matrix
    """
    
    for i in range(1):
        for j in range(1):
            start = time()
            seeds = [20,20,70,10,20,20,10,70]
            alpha = 0.1
            sigma = 50
            print()
            print("Parameters: alpha = {}, sigma = {}".format(alpha, sigma))
            G = create_graph(data, alpha, sigma, seeds)
            cut_value, partition = nx.minimum_cut(G, "s", "t")
            plot_image(data, partition)
            
            end= time()
            time_elapsed = end - start
            print("Time Elapsed Discrete: ", time_elapsed)

def cont_segment(data):
    """Continuous Segmentation: MBMF ALgorithm
        
    Inputs:
        data: the image data matrix
    """
    
    start = time()
    data_cont = np.matrix(data)/255
    
    """
    param 0 - the maximum number of iterations
    param 1 - the error bound for convergence
    param 2 - cc for the step-size of augmented Lagrangian method
    param 3 - the step-size for the graident-projection of p
    """
    params = [500, 1e-4, 0.3, 0.25]
    l = 0.5
    λ, erriter, i = CMF_2D(data_cont, params, [0.4,1], α = 0.2)
    
    print("Iterations: {}".format(i))
    print("Erriter mean: {}".format(np.mean(erriter)))
    
    plot_2d(λ, data_cont, l)
    end= time()
    time_elapsed = end - start
    print("Time Elapsed Continuous: ", time_elapsed)

""" Input the file name and scaling factor. This then runs the algorithms"""
file_name = "test_image"
scale = 0.1

print("Discrete Image Segmentation:")
print()
img_data_col(file_name, scale)
data_col = open_csv_col(file_name)
discrete_segment(data_col)
print()
print()

print("Continuous Image Segmentation")
print()
img_data(file_name, scale)
data_bw = open_csv(file_name)
cont_segment(data_bw)


