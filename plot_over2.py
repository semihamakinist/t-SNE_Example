# -*- coding: utf-8 -*-
"""
Created on Tue Jan 08 12:52:45 2019

@author: semiha
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 08 12:00:02 2019

@author: semiha
"""

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os

main_path = os.getcwd()
sample_dir = os.path.join(main_path, "kelimeVec-1.txt")

arr_x = [] 
arr_y = [] 

norm = plt.Normalize(1, 4)
cmap = plt.cm.RdYlGn
c=np.array([3, 3, 3, 1, 2, 2, 1, 1, 1])

def update_annot(ind):

    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}, {}".format(" ".join(list(map(str, ind["ind"]))), 
                           " ".join([arr_y[n] for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()
                
if __name__ == "__main__":
    
    with open(sample_dir, "r") as ins:
        for line in ins:
            values = line.strip().split()
            arr_y.append(values[0])
            arr_x.append(values[1:])

    model = TSNE(learning_rate = 200)
    
    transformed = model.fit_transform(arr_x)
    
    xs = transformed[:,0]
    ys = transformed[:,1]
    
    #set plot    
    fig, ax = plt.subplots()
    species = [0, 0, 0, 1, 2, 2, 1, 1, 1]
    sc = plt.scatter(xs, ys, c=c, s=100, cmap=cmap, norm=norm)
    
    
    annot = ax.annotate("", xy=(0,0), xytext=(-20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    
    fig.canvas.mpl_connect("motion_notify_event", hover)
    
    plt.show()