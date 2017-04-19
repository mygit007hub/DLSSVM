import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import random
import matplotlib.patches as patches


global w

def load_image(frame_id):
    filename = "../Basketball/img/%04d.jpg" % frame_id
    img = io.imread(filename)
    return img

    
def feature_vector(X,Y):
    lab = color.rgb2lab(X)
    gray = color.rgb2gray(X)
    return np.reshape(np.append(lab,gray),(np.shape(X)[0],np.shape(X)[1],np.shape(X)[2]+1))


def show_img(img,border):
    fig = plt.figure()
    rect = fig.add_subplot(111, aspect='equal')
    rect.add_patch(
        patches.Rectangle(
            (border[0],border[1]),  # (x,y)
            border[2],      # width
            border[3],      # height
            fill=False,     # remove background
            edgecolor="red" # edgecolor = "#0000ff"
        )
    )
    plt.imshow(img)
    plt.show()
    return 

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

            
% yi needed to upated is placed in the end of support vectors
sampleID = patterns{idPat}.supportVectorNum(n);

H = patterns{idPat}.lossY(sampleID) - w0 * patterns{idPat}.X(sampleID, :)';
% g_ij

alpha_old = patterns{idPat}.supportVectorAlpha(n);
% alpha_ij

s = params.lambda - (sum(patterns{idPat}.supportVectorAlpha) - alpha_old);
% 1 - alpha_i + alpha_old

kerProduct = patterns{idPat}.X(sampleID,:) * patterns{idPat}.X(sampleID, :)';
% h_ij = x_ij' * x_ij

d = H / kerProduct;
% g_ij / h_ij

alpha = min(max(alpha_old + d, 0), s);
% Original:    alpha = alpha_old + min(max(-alpha_i, d), 1 - alpha_i)
% Derivation:  alpha = min(max(-alpha_i, d) + alpha_old, 1 - alpha_i + alpha_old)
%                    = min(max(-alpha_i, d) + alpha_old, s)
%                    = min(max(alpha_old + d, alpha_old - alpha_i), s)

d = alpha - alpha_old;
% d is alpha_star

w0 = w0 + d * patterns{idPat}.X(sampleID, :);

weight = alpha * alpha * kerProduct;

if alpha == 0
    [patterns, deletePat] = svBudgetMaintain_zeros(patterns, idPat, sampleID);
    if deletePat == 1
        return;
    end
else
    patterns{idPat}.supportVectorAlpha(n) = alpha;
    patterns{idPat}.supportVectorWeight(n) = weight;
end