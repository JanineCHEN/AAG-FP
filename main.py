###### EXTRACT NETWORKX GRAPHS BASED ON FLOORPLAN IMAGES
# -*- coding: utf-8 -*-
"""
Created on Feb 19 2021

@author: CHEN JIELIN
"""

import os
import time
import numpy as np
import pandas as pd
import networkx as nx

from graph_extractor import graph_extractor
from methods import *

out_dir = './outputs'
img_dir = './FP_sample_images'

for file in os.listdir(img_dir):
    filename = os.fsdecode(file)
    if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
        print(filename) 

        out_path = os.path.join(out_dir, filename.split('.')[-2])
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        img_path = os.path.join(img_dir, filename)
        G,G_n,G_m = graph_extractor(img_path, out_path)

        draw_img(img_path,out_path)
        draw_simple_graph(G_n,out_path)
        draw_multigraph(G_m,out_path)

        nx.write_gpickle(G, os.path.join(out_path,"G.gpickle"))
        nx.write_gpickle(G_n, os.path.join(out_path,"G_n.gpickle"))
        nx.write_gpickle(G_m, os.path.join(out_path,"G_m.gpickle"))
    
    else:
        print("Could not find floor plan images in the folder")
