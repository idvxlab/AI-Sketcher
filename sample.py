
# import the required libraries# impor 
import numpy as np
import time
import random
#import cPickle
import pickle
import codecs
import collections
import os
import math
import json
import tensorflow as tf
from six.moves import xrange

# libraries required for visualisation:
# from IPython.display import SVG, display
import svgwrite # conda install -c omnia svgwrite=1.1.6
import PIL
from PIL import Image
import matplotlib.pyplot as plt

# set numpy output to something sensible
np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)

# import our command line tools
import sys
sys.path.append('./py/')
from train import *
from model import *
from utils import *
from rnn import *

# little function that displays vector images and saves them to .svg
def draw_strokes(data, factor=0.2, svg_filename = 'sample.svg'):
  tf.gfile.MakeDirs(os.path.dirname(svg_filename))
  min_x, max_x, min_y, max_y = get_bounds(data, factor)
  dims = (50 + max_x - min_x, 50 + max_y - min_y)
  dwg = svgwrite.Drawing(svg_filename, size=dims)
  #add white background
  dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))
  lift_pen = 1
  abs_x = 25 - min_x 
  abs_y = 25 - min_y
  p = "M%s,%s " % (abs_x, abs_y)

  # control m to resample
  command = "m"
  for i in range(len(data)):
    if (lift_pen == 1):
      command = "m"
    elif (command != "l"):
      command = "l"
    else:
      command = ""
    x = float(data[i,0])/factor
    y = float(data[i,1])/factor
    lift_pen = data[i, 2]
    p += command+str(x)+","+str(y)+" "
  the_color = "black"
  stroke_width = 1
  dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
  dwg.save()
  #display(SVG(dwg.tostring()))

# generate a 2D grid of many vector drawings
def make_grid_svg(s_list, grid_space=10.0, grid_space_x=16.0):
  def get_start_and_end(x):
    x = np.array(x)
    x = x[:, 0:2]
    x_start = x[0]
    x_end = x.sum(axis=0)
    x = x.cumsum(axis=0)
    x_max = x.max(axis=0)
    x_min = x.min(axis=0)
    center_loc = (x_max+x_min)*0.5
    return x_start-center_loc, x_end
  x_pos = 0.0
  y_pos = 0.0
  result = [[x_pos, y_pos, 1]]
  for sample in s_list:
    s = sample[0]
    grid_loc = sample[1]
    grid_y = grid_loc[0]*grid_space+grid_space*0.5
    grid_x = grid_loc[1]*grid_space_x+grid_space_x*0.5
    start_loc, delta_pos = get_start_and_end(s)

    loc_x = start_loc[0]
    loc_y = start_loc[1]
    new_x_pos = grid_x+loc_x
    new_y_pos = grid_y+loc_y
    result.append([new_x_pos-x_pos, new_y_pos-y_pos, 0])

    result += s.tolist()
    result[-1][2] = 1
    x_pos = new_x_pos+delta_pos[0]
    y_pos = new_y_pos+delta_pos[1]
  return np.array(result)


def draw_svg(n,stroke,sess,sample_model,eval_model,pre_hps_model,c_n,user_stroke=0, z_input=None):


  c = [0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 
      0,0,0,0,0,0,0, 0,0,0,0,0,0,0,
      0,0,0,0,0,0,0, 0,0,0,0,0,0,0]

  c[n] = c_n+0.0001

  stroke_len = 0
  resample = True

  def encode(input_strokes, c):
    print '1'
    print(sess)
  #strokes = to_normal_strokes(input_strokes)
    strokes = input_strokes
    strokes = np.array(eval(strokes)) #.tolist()
    l = strokes.shape[0]

    strokes = strokes.tolist()

    for i in range(273-l):
      strokes.insert(0, [0, 0, 1, 0, 0])

    seq_len = [len(input_strokes)]
    c = c[0]
    c = c[:14]

    return sess.run(eval_model.batch_z, feed_dict={eval_model.input_data: [strokes], eval_model.sequence_lengths: [l], eval_model.c_expr:[c]})[0]

  def decode(p, c, z_input=None, draw_mode=True, temperature=10, factor=0.2):
    print(c)
    z = None
    if z_input is not None:
      z = [z_input]
    sample_strokes, _ = sample(sess, p, sample_model, seq_len=eval_model.hps.max_seq_len, temperature=temperature, greedy_mode=True, z=z, c=c)

    return sample_strokes

  if user_stroke == 0:
    z_input = encode(stroke,[c])
    stroke_result = decode(p=pre_hps_model, c=[c], z_input=z_input, temperature=0.01, draw_mode=False)[0]
    stroke_len = len(stroke_result)

  else:    
    while resample:
      z_input = encode(stroke,[c])
      stroke_result = decode(p=pre_hps_model, c=[c], z_input=z_input, temperature=0.01, draw_mode=False)[0]
      stroke_len = len(stroke_result)
      if (stroke_len-user_stroke)>=-5 and (stroke_len-user_stroke)<=5:
        resample = False
         
  return stroke_result
  # queue.put(stroke_result.tolist())






