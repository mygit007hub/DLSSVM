import numpy as np
import matplotlib as plt
from skimage import io, color
import random

global w
def load_image(frame_id):
    filename = "../Basketball/img/%04d.jpg" % frame_id
    img = io.imread(filename)
    lab = color.rgb2lab(rgb)
    return img

def feature_vector(X,Y):
    pass

def score(w,X,Y):
    return w.T.dot(feature_vector(X,Y))

def loss_function():
    pass

#  Input x = feature_vector(x,y) - feature_vector(x,h)  ;   l = loss_function(y,h)
def Optimize(x,l) :
    aij,ai,w = [0,0,0]  # Initialize variables (if not passed as arguments)
    while True :
        aij = random.uniform(1.5, 1.9) # Randomly pick a dual variable α_ij ;
        gij = l - wT*x                 # Compute gradient gij = lij - w'*x_ij from (11);
        hij = -xTx
        if gij > err and ai ==1 :   # Find another variable if linear constraint is activ
            a2 = random.uniform(1.5, 1.9)   # Randomly pick another dual variable with same i (α_ik for k != j);
            a_star = min(max(-max(aij,1-aik),gij/hij),min(aik,1-aij)) # Compute a^∗ with (15) ;
            # Update αij , αik, w with (16),(17)
            aij = aij + a_star
            aik = aik + a_star
            w = w + a_star * xij
        elif abs(gij) > e :         #/* Else update single dual variable */
            a_star = min(max(-ai,gij/hij),1-ai)
            aij = aij+a_star
            ai = ai+ a_star
            w = w+ a_star*xij