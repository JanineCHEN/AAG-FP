###### EXTRACT NETWORKX GRAPHS BASED ON FLOORPLAN IMAGES
# -*- coding: utf-8 -*-
"""
Created on Feb 19 2021

@author: CHEN JIELIN
"""

import fiona
import os
from affine import Affine
from rasterio import features
import numpy as np
import networkx as nx
import shapely
from shapely.geometry import Polygon as sPolygon
from shapely.geometry import shape
from shapely.geometry import LineString as sLine
from shapely.geometry import Point as sPoint
from shapely.geometry import MultiPoint, MultiPolygon,MultiLineString
from shapely.strtree import STRtree
import geopandas as gpd

import cv2
import PIL
import matplotlib.pyplot as plt
from PIL import Image

from collections import Counter
from utils import *

def wall_instances_simplification(wall_instances,height,width,k_size=15,scale_1=1.03,scale_2=1.0):

    '''
    Using list of wall instances as input, vectorize the wall instances and convert to rectangles
    Extend the rectangles to nearby ones to close up the gaps
    wall_instances: extracted using Mask R-CNN models
    k_size: control the strength of thicken the wall instances, default is 15
    scale_1: control extension scale of the rectangle alongside the long axis before closing up the gaps, default is 1.03
    scale_2: control extension scale of the final rectangle alongside the long axis, default is 1.0
    '''
    # vectorize wall instances extracted by Mask R-CNN
    kernel = np.ones((k_size,k_size), np.uint8)

    #get vectorized shapes using rasterio
    rectangles = []
    long_axises = []
    short_axises = []
    aff = Affine(1, 0, 1, 0, -1, 0) # Adjust flip and symmetry
    h = height
    w = width
    gap = max(h,w) * 0.015

    for wall_i in wall_instances:

        wall = cv2.dilate(wall_i, kernel,iterations = 1)
        wall = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, kernel,iterations = 1)
        wall = (wall != 0).astype(int)
        mask = wall == 1
        shape_wall = features.shapes(wall.astype('uint8'), mask=mask, transform = aff)

        for s, v in shape_wall:
            ## polygon without holes
            if np.array(s['coordinates']).shape[0] == 1:
                polygon = sPolygon(np.array(s['coordinates']).squeeze())
            ## polygon with holes
            if np.array(s['coordinates']).shape[0] != 1:
                exterior = np.array(np.array(s['coordinates'])[0]).squeeze()
                interior = [np.array(p).squeeze() for p in np.array(s['coordinates'])[1:]]
                polygon = sPolygon(exterior, interior)

        # center = polygon.centroid
        # get all the boundary points
        xy = np.array(polygon.exterior.coords.xy)
        ## geometric center of the polygon
        center = xy.mean(axis=-1)
        # get the general minimum bounding rectangle that contains the object. Unlike envelope this rectangle is not constrained to be parallel to the coordinate axes.
        rectangle = MultiPoint(xy.T).minimum_rotated_rectangle
        # find the eigenvectors of the covariance matrix of the point cloud. The aspect ratio is the ratio of the largest to smallest eigenvalues
        eigvals, eigvecs = np.linalg.eig(np.cov(xy))
        # https://stackoverflow.com/questions/7059841/estimating-aspect-ratio-of-a-convex-hull
        if eigvals[0] < eigvals[1]:
            val = eigvals[0] * 5
            vec = eigvecs.T[0]
            x,y = np.vstack((center + val * vec, center - val * vec)).T
            minor_axis = sLine([(x[0],y[0]),(x[1],y[1])])
            val = eigvals[1] * 2
            vec = eigvecs.T[1]
            x,y = np.vstack((center + val * vec, center - val * vec)).T
            major_axis = sLine([(x[0],y[0]),(x[1],y[1])])
        if eigvals[0] > eigvals[1]:
            val = eigvals[0] * 2
            vec = eigvecs.T[0]
            x,y = np.vstack((center + val * vec, center - val * vec)).T
            major_axis = sLine([(x[0],y[0]),(x[1],y[1])])
            val = eigvals[1] * 5
            vec = eigvecs.T[1]
            x,y = np.vstack((center + val * vec, center - val * vec)).T
            minor_axis = sLine([(x[0],y[0]),(x[1],y[1])])
        # get the intersection points of the axis and polygon(rectangle)
        long_line = sLine(np.array(major_axis.intersection(rectangle).coords.xy).T)
        short_line = sLine(np.array(minor_axis.intersection(rectangle).coords.xy).T)

        # Extend the rectangle alongside the long axis
        long_line_e = shapely.affinity.scale(long_line, xfact=scale_1, yfact=scale_1, zfact=scale_1)

        # create a new rectangle using the extended long axis and the short axis
        rectangle_e = long_line_e.buffer(short_line.length/6)

        # check if the rectangle already exist in the list (overlap with existing rectangles)
        add = 0
        if rectangles:
            for rec in rectangles:
                if rectangle_e.intersects(rec):
                    overlap = rectangle_e.intersection(rec).area/rectangle_e.area
                    if overlap > 0.35:
                        add += 1
                        continue
        if add != 0: # repeated wall
            continue

        # check the nearest rectangles of the new rectangle in the list
        if add == 0:
            if long_axises:
                for line in long_axises:
                    p0_list = [sPoint(long_line_e.coords[0]),sPoint(long_line_e.coords[1])]
                    distance = max(h,w)
                    p1_list = [sPoint(line.coords[0]),sPoint(line.coords[1])]
                    if distance > p0_list[0].distance(p1_list[0]):
                        distance = p0_list[0].distance(p1_list[0])
                        p_idx = [0,0]
                    if distance > p0_list[0].distance(p1_list[1]):
                        distance = p0_list[0].distance(p1_list[1])
                        p_idx = [0,1]
                    if distance > p0_list[1].distance(p1_list[1]):
                        distance = p0_list[1].distance(p1_list[1])
                        p_idx = [1,1]
                    if distance > p0_list[1].distance(p1_list[0]):
                        distance = p0_list[1].distance(p1_list[0])
                        p_idx = [1,0]
                    if distance < gap and distance != 0:
                        p1 = sLine([p0_list[p_idx[0]],p1_list[p_idx[1]]]).centroid
                        if p_idx[0] == 0:
                            p2 = p0_list[1]
                        if p_idx[0] == 1:
                            p2 = p0_list[0]
                        long_line_e = sLine([p1,p2])

            long_line_e = shapely.affinity.scale(long_line_e, xfact=scale_2, yfact=scale_2, zfact=scale_2)
            long_axises.append(long_line_e)
            rectangle_e = long_line_e.buffer(short_line.length/6)
            rectangles.append(rectangle_e)

    return rectangles


def wall_smooth(wall,crack_triangle_ratio):

    crack_triangle = wall.area * crack_triangle_ratio

    if wall.boundary.geom_type == 'LineString':
        is_simple = True
        p_list = list(wall.boundary.coords)
        p_list_s = []
        j = 0
        k = 2
        for i in range(len(p_list)):
            if k == len(p_list):
                break
            triangle = sPolygon([sPoint(p_list[j]),sPoint(p_list[k]),sPoint(p_list[k-1])])
            c = triangle.centroid
            if c.within(wall): # c.within(wall) # wall.contains(c)
                p_list_s.append(p_list[k-1])
            if not c.within(wall):
                if triangle.area > crack_triangle:
                    p_list_s.append(p_list[k-1])
            j += 1
            k += 1

        wall_s = sPolygon(p_list_s)

    if wall.boundary.geom_type == 'MultiLineString':
        is_simple = False
        p_list = list(spline.coords for spline in wall.boundary.geoms)
        # deal with exterior
        p_list_s_ex = []
        j = 0
        k = 2
        for i in range(len(p_list[0])):
            if k == len(p_list[0]):
                break
            triangle = sPolygon([sPoint(p_list[0][j]),sPoint(p_list[0][k]),sPoint(p_list[0][k-1])])
            c = triangle.centroid
            if c.within(wall):
                p_list_s_ex.append(p_list[0][k-1])
            if not c.within(wall):
                if triangle.area > crack_triangle:
                    p_list_s_ex.append(p_list[0][k-1])
            j += 1
            k += 1
        # deal with interior
        p_list_s_in = []
        for n in range(len(p_list)-1):
            p_list_s_in_ = []
            j = 0
            k = 2
            for i in range(len(p_list[n+1])):
                if k == len(p_list[n+1]):
                    break
                triangle = sPolygon([sPoint(p_list[n+1][j]),sPoint(p_list[n+1][k]),sPoint(p_list[n+1][k-1])])
                c = triangle.centroid
                if c.within(wall):
                    p_list_s_in_.append(p_list[n+1][k-1])
                if not c.within(wall):
                    if triangle.area > crack_triangle:
                        p_list_s_in_.append(p_list[n+1][k-1])
                j += 1
                k += 1
            if p_list_s_in_ and len(p_list_s_in_) > 2:
                p_list_s_in.append(p_list_s_in_)

        exterior = p_list_s_ex
        interior = p_list_s_in
        wall_s = sPolygon(exterior, interior)

    return wall_s, is_simple


def vectorization(img, output_path, img_dir, min_area = 20, min_room = 5000):
    '''
    # Parameters: img, output_path, img_dir, min_area = 5
    - img: Pre-processed parsed floorplan image, type uint8, binary
    - output_path: A directory to store pre-processed image and output polygon files
    - img_dir: Original image directory
    - min_area: a threshold parameter to remove small polygons
    # Usage
    Input the target image to vectorize by using Rasterio lib.
    This function has several steps inbetween, like bufferizing polygons, filling the bubbles \
    (inner holes in/ or between polygons) by making them into polygon.
    # output
    A geodataframe extracted from the image
    '''
    filename = img_dir.split('/')[-1]
    #get vectorized shapes using rasterio
    aff = Affine(1, 0, 1, 0, -1, 0) # Adjust flip and symmetry
    outside = img == 0
    walls = img == 1
    rooms = img == 2
    doors = img == 3
    windows = img == 4
    stairs = img == 5
    shapes_outside = features.shapes(img, mask=outside, transform = aff)
    shapes_walls = features.shapes(img, mask=walls, transform = aff)
    shapes_rooms = features.shapes(img, mask=rooms, transform = aff)
    shapes_doors = features.shapes(img, mask=doors, transform = aff)
    shapes_windows = features.shapes(img, mask=windows, transform = aff)
    shapes_stairs = features.shapes(img, mask=stairs, transform = aff)
    results_shapes_outside = ({'properties': {'obj_class': 0, 'class_name': 'outside'}, 'geometry': s} for i, (s, v) in enumerate(shapes_outside))
    results_shapes_walls = ({'properties': {'obj_class': 1, 'class_name': 'walls'}, 'geometry': s} for i, (s, v) in enumerate(shapes_walls))
    results_shapes_rooms = ({'properties': {'obj_class': 2, 'class_name': 'rooms'}, 'geometry': s} for i, (s, v) in enumerate(shapes_rooms))
    results_shapes_doors = ({'properties': {'obj_class': 3, 'class_name': 'doors'}, 'geometry': s} for i, (s, v) in enumerate(shapes_doors))
    results_shapes_windows = ({'properties': {'obj_class': 4, 'class_name': 'windows'}, 'geometry': s} for i, (s, v) in enumerate(shapes_windows))
    results_shapes_stairs = ({'properties': {'obj_class': 5, 'class_name': 'stairs'}, 'geometry': s} for i, (s, v) in enumerate(shapes_stairs))
    #save polygon in shape/GeoJSON format
    with fiona.open(
            os.path.join(output_path,filename.split('.')[-2]+'_polys.shp'), 'w',
            driver='ESRI Shapefile',
            schema={'properties': [('obj_class', 'int'), ('class_name', 'str')],
                    'geometry': 'Polygon'}) as dst:
        dst.writerecords(results_shapes_outside)
        dst.writerecords(results_shapes_walls)
        dst.writerecords(results_shapes_rooms)
        dst.writerecords(results_shapes_doors)
        dst.writerecords(results_shapes_windows)
        dst.writerecords(results_shapes_stairs)
    df = gpd.read_file(os.path.join(output_path,filename.split('.')[-2]+'_polys.shp'))
    # print(df)
    # remove useless and small polygons whose area is less than threshold
    to_pop = [i for i in range(len(df)) if df.geometry.iloc[i].area < min_area]
    df = df.drop(df.index[to_pop])
    df = df.reset_index(drop = True)
    # print(df)
    # remove small false rooms
    to_pop = [i for i in range(len(df)) if df.obj_class.iloc[i] == 2 and df.geometry.iloc[i].area < min_room]
    df = df.drop(df.index[to_pop])
    df = df.reset_index(drop = True)
    # print(df)
    #export processed polygon geodataframe
    df.to_file(os.path.join(output_path,filename.split('.')[-2]+"_polys.shp"))


def vectorization_mask(img):

    h,w = img.shape[:2]

    #get vectorized shapes using rasterio
    aff = Affine(1, 0, 1, 0, -1, 0) # Adjust flip and symmetry
    mask = img == 255
    
    if True in np.unique(mask):
        shapes = features.shapes(img, mask=mask, transform = aff)

        area = 0
        i = 0
        for s, v in shapes:
            ## polygon without holes
            if np.array(s['coordinates']).shape[0] == 1:
                polygon = sPolygon(np.array(s['coordinates']).squeeze())
                if polygon.area > area:
                    area = polygon.area
                    mask_background_idx = i
                i += 1
            ## polygon with holes
            if np.array(s['coordinates']).shape[0] != 1:
                exterior = np.array(np.array(s['coordinates'])[0]).squeeze()
                interior = [np.array(p).squeeze() for p in np.array(s['coordinates'])[1:]]
                polygon = sPolygon(exterior, interior)
                # for p in np.array(s['coordinates']):
                    # polygon = Polygon(np.array(p).squeeze())
                if polygon.area > area:
                    area = polygon.area
                    mask_background_idx = i
                i += 1

        mask_area = area

        shapes = features.shapes(img, mask=mask, transform = aff)
        mask_background = features.rasterize(
                ((g, 255) for i, (g, v) in enumerate(shapes) if i == mask_background_idx), 
                all_touched=True, 
                out_shape=(h,w),
                transform = aff,
                )
        
    if not True in np.unique(mask):
        mask_background = np.zeros_like(img)
        mask_area = 0

    return mask_background, mask_area


def build_FPgraph_RAG(input_path, output_path, img_dir):

    # make a region adjacency graph for vector data which only has polygons
    filename = img_dir.split('/')[-1]
    df = gpd.read_file(os.path.join(input_path, filename.split('.')[-2]+'_polys.shp'))

    nodes = dict()

    for i in range(len(df)):
        curpoly = df.iloc[i]['geometry']
        obj_class = df.iloc[i]['obj_class']
        class_name = df.iloc[i]['class_name']
        nodes[i] = {'polygon':curpoly, 'point' : (curpoly.centroid.x, curpoly.centroid.y), 'area' : curpoly.area, 'class_id' : obj_class, 'class_name' : class_name}
    
    # make RAG
    tree = STRtree(df['geometry'].buffer(1, 10, cap_style = 2, join_style = 2))
    n_sum = 0
    n_list = []
    poly_dict = dict()
    pidx = 0

    for i in df['geometry'].buffer(1, 10, cap_style = 2, join_style = 2):
        poly_dict[i.bounds] = pidx
        pidx += 1

    for i in range(len(df)):
        q_poly = df['geometry'][i].buffer(1, 10, cap_style = 2, join_style = 2)
        try:
            curnei = [p_poly for p_poly in tree.query(q_poly) if p_poly.intersects(q_poly) and p_poly != q_poly]
            cur_list = [poly_dict[i.bounds] for i in curnei]
            n_sum += len(curnei)
            n_list.append(cur_list)
        except:
            n_list.append([])

    edges = dict()
    eidx = 0

    for i in range(len(n_list)):
        for j in n_list[i]:
            curtuple = (i, j)
            if curtuple not in edges.values() and (curtuple[1], curtuple[0]) not in edges.values() and curtuple[0] != curtuple[1]:
                edges[eidx] = curtuple
                eidx += 1
    
    G = nx.Graph()
    for i in nodes:
        G.add_node(i, polygon=nodes[i]['polygon'], point=nodes[i]['point'], area=nodes[i]['area'], class_id=nodes[i]['class_id'],class_name=nodes[i]['class_name'])

    G.add_edges_from(edges.values())
    
    # construct edge lines as shapely objs and export it in .shp format file    
    lines = gpd.GeoDataFrame(columns = ['eidx', 'polygon1', 'class1', 'name1', 'polygon2', 'class2', 'name2', 'geometry'])
    eidx = 0
    for i in edges.values():
        p1 = nodes[i[0]]
        p2 = nodes[i[1]]
        class_1 = p1['class_id']
        class_2 = p2['class_id']
        name_1 = p1['class_name']
        name_2 = p2['class_name']
        point1 = sPoint(p1['point'])
        point2 = sPoint(p2['point'])
        curline = sLine([point1, point2])
        lines = lines.append({'eidx':eidx, 'polygon1': i[0], 'class1': class_1, 'name1': name_1, 'polygon2':i[1], 'class2': class_2, 'name2': name_2, 'geometry':curline}, ignore_index = True)
        eidx += 1
   
    lines.to_file(os.path.join(input_path, filename.split('.')[-2]+"_adjacency.shp"))
    return G, nodes, edges


def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def draw_img(img_dir,output_path):
    img = Image.open(img_dir)
    im = img.convert('RGB')
    im = np.array(im)
    h,w=im.shape[:2]

    plt.figure(figsize=(w/160,h/160))
    plt.axis('off')
    plt.imshow(im)
    plt.savefig(os.path.join(output_path,'FP.jpg'))


def draw_simple_graph(G_n,output_path):
    labeldict = {}
    for i in range(len(G_n.nodes)):
        if G_n.nodes[list(G_n.nodes)[i]]['class_name'] == 'rooms':
            labeldict[list(G_n.nodes)[i]] = f"{G_n.nodes[list(G_n.nodes)[i]]['class_name']}_{int(round(G_n.nodes[list(G_n.nodes)[i]]['area_ratio']*100))}%"
        else:
            labeldict[list(G_n.nodes)[i]] = G_n.nodes[list(G_n.nodes)[i]]['class_name']

    posdict = {}
    for i in range(len(G_n.nodes)):
        if G_n.nodes[list(G_n.nodes)[i]]['class_name'] == 'outside':
            posdict[list(G_n.nodes)[i]] = (0,0)
        # elif G_n.nodes[list(G_n.nodes)[i]]['class_name'] == 'stairs':
            # posdict[list(G_n.nodes)[i]] = (0,-h)
        else:    
            posdict[list(G_n.nodes)[i]] = G_n.nodes[list(G_n.nodes)[i]]['point']

    nodes_color = []
    for i in range(len(G_n.nodes)):
        if G_n.nodes[list(G_n.nodes)[i]]['class_name'] == 'outside':
            nodes_color.append('#CCCCCC')
        if G_n.nodes[list(G_n.nodes)[i]]['class_name'] == 'rooms':
            nodes_color.append('#F6D55C')
        if G_n.nodes[list(G_n.nodes)[i]]['class_name'] == 'stairs':
            nodes_color.append('#ED553B')

    edges_color = []
    for i in G_n.edges:
        if G_n.edges[i]['edge_class'] == 'window':
            edges_color.append('#3CAEA3')
        if G_n.edges[i]['edge_class'] == 'door':
            edges_color.append('#F67E7D')
        if G_n.edges[i]['edge_class'] == 'direct':
            edges_color.append('#173F5F')
        if G_n.edges[i]['edge_class'] == 'wall':
            edges_color.append('#AAAAAA')

    edges_label = {}
    for i in G_n.edges:
        edges_label[i] = G_n.edges[i]['edge_class']

    plt.figure(figsize=(12,7))
    nx.draw(G_n, labels=labeldict, with_labels = True, pos = posdict, width= 3, node_size=2000, node_color = nodes_color, edge_color = edges_color, font_weight='bold')
    nx.draw_networkx_edge_labels(G_n,pos = posdict,
                                edge_labels=edges_label,
                                font_color='black')
    plt.axis('off')
    plt.savefig(os.path.join(output_path,'simplegraph.png'))

def draw_multigraph(G_m,output_path):
    labeldict = {}
    for i in range(len(G_m.nodes)):
        if G_m.nodes[list(G_m.nodes)[i]]['class_name'] == 'rooms':
            labeldict[list(G_m.nodes)[i]] = f"{G_m.nodes[list(G_m.nodes)[i]]['class_name']}_{int(round(G_m.nodes[list(G_m.nodes)[i]]['area_ratio']*100))}%"
        else:
            labeldict[list(G_m.nodes)[i]] = G_m.nodes[list(G_m.nodes)[i]]['class_name']

    posdict = {}
    for i in range(len(G_m.nodes)):
        if G_m.nodes[list(G_m.nodes)[i]]['class_name'] == 'outside':
            posdict[list(G_m.nodes)[i]] = (0,0)
        # elif G_m.nodes[list(G_m.nodes)[i]]['class_name'] == 'stairs':
            # posdict[list(G_m.nodes)[i]] = (0,-h)
        else:    
            posdict[list(G_m.nodes)[i]] = G_m.nodes[list(G_m.nodes)[i]]['point']

    nodes_color = []
    for i in range(len(G_m.nodes)):
        if G_m.nodes[list(G_m.nodes)[i]]['class_name'] == 'outside':
            nodes_color.append('#CCCCCC')
        if G_m.nodes[list(G_m.nodes)[i]]['class_name'] == 'rooms':
            nodes_color.append('#F6D55C')
        if G_m.nodes[list(G_m.nodes)[i]]['class_name'] == 'stairs':
            nodes_color.append('#ED553B')

    plt.figure(figsize=(12,7))
    ax = plt.gca()
    i = 0
    for e in G_m.edges:
        if list(G_m.edges.data())[i][-1]['edge_class'] == 'wall':
            ax.annotate("",
                        xy=posdict[e[0]], xycoords='data',
                        xytext=posdict[e[1]], textcoords='data',
                        arrowprops=dict(arrowstyle="wedge", # wedge
                                        color="#AAAAAA",
                                        shrinkA=25, shrinkB=25,
                                        patchA=None, patchB=None,
                                        connectionstyle="arc3,rad=rrr".replace('rrr',str(0.05*e[2])),
                                        ),
                        )
        if list(G_m.edges.data())[i][-1]['edge_class'] == 'window':
            ax.annotate("",
                        xy=posdict[e[0]], xycoords='data',
                        xytext=posdict[e[1]], textcoords='data',
                        arrowprops=dict(arrowstyle="wedge", # wedge
                                        color="#3CAEA3",
                                        shrinkA=25, shrinkB=25,
                                        patchA=None, patchB=None,
                                        connectionstyle="arc3,rad=rrr".replace('rrr',str(0.05*e[2])),
                                        ),
                        )
        if list(G_m.edges.data())[i][-1]['edge_class'] == 'door':
            ax.annotate("",
                        xy=posdict[e[0]], xycoords='data',
                        xytext=posdict[e[1]], textcoords='data',
                        arrowprops=dict(arrowstyle="wedge", # wedge
                                        color="#F67E7D",
                                        shrinkA=25, shrinkB=25,
                                        patchA=None, patchB=None,
                                        connectionstyle="arc3,rad=rrr".replace('rrr',str(0.05*e[2])),
                                        ),
                        )
        if list(G_m.edges.data())[i][-1]['edge_class'] == 'direct':
            ax.annotate("",
                        xy=posdict[e[0]], xycoords='data',
                        xytext=posdict[e[1]], textcoords='data',
                        arrowprops=dict(arrowstyle="wedge", # wedge
                                        color="#173F5F",
                                        shrinkA=25, shrinkB=25,
                                        patchA=None, patchB=None,
                                        connectionstyle="arc3,rad=rrr".replace('rrr',str(0.05*e[2])),
                                        ),
                        )
        i += 1
    nx.draw(G_m, labels=labeldict, with_labels = True, pos = posdict, width= 0, node_size=2000, node_color = nodes_color, font_weight='bold')
    plt.axis('off')
    plt.savefig(os.path.join(output_path,'multigraph.png'))
