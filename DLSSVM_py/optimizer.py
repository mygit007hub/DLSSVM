import numpy as np
import math

P = 5
Q = 10
sv_max = 10
thr = 0.0001

class Optimizer:
    def __init__(self):
        self.frame_id = 0
        self.patterns = {}
        self.w = 0
    
    def new_frame(self, frame_id, sampler):
        self.frame_id = frame_id
        self.sampler = sampler
    
    def optimize(self, target):
        self._add_pattern(target)
        for p in range(P):
            n = len(self.patterns) - 1
            k_id = n - math.floor(p * n / P)
            id = list(self.patterns.keys())[k_id]
            self._update_working_set(id)
            self._update_alpha(id)
            self._maintain_sv_budgets()
            for q in range(Q):
                k_id = n - math.floor(q * n / Q)
                id = list(self.patterns.keys())[k_id]
                self._update_alpha(id)

    def predict(self, target):
        (X, Y) = self.sampler.get_samples()
        score = np.inner(self.w, X)
        max_i = np.argmax(score)
        return Y[max_i]
    
    def _add_pattern(self, target):
        (X, Y) = self.sampler.get_samples()
        Xi = self.sampler.crop(target)
        Yi = target
        dX = Xi - X
        self.patterns[self.frame_id] = Pattern(Xi, Yi, dX, Y)
        if not self.w:
            self.w = np.zeros(Xi.shape)
    
    # Find SupportVector
    def _update_working_set(self, id):
        score = self.patterns[id].loss - self.w.dot(self.patterns[id].dX.T)
        i = np.argmax(score)
        if i not in self.patterns[id].sv:
            self.patterns[id].sv[i] = SupportVector()

    # DCD(Dual Coordinate Descent) optimization
    def _update_alpha(self, id):
        if len(self.patterns[id].sv):
            sv_n = list(self.patterns[id].sv.keys())
            if sum(self.w) == 0:
                score = 1
            else:
                score = self.patterns[id].loss[sv_n] - self.w.dot(self.patterns[id].dX[sv_n].T)
            n = sv_n[np.argmax(score)]
            g_ij = self.patterns[id].loss[n] - self.w.dot(self.patterns[id].dX[n])
            h_ij = self.patterns[id].dX[n].dot(self.patterns[id].dX[n].T) + 10e-8
            alpha = self.patterns[id].sv[n].alpha
            alpha_i = sum([self.patterns[id].sv[j].alpha for j in self.patterns[id].sv])
            alpha_star = min(max(-alpha_i, g_ij / h_ij), 1 - alpha_i)
            alpha = alpha + alpha_star
            weight = alpha * alpha * h_ij
            self.w = self.w + alpha_star * self.patterns[id].dX[n]
            if alpha:
                self.patterns[id].sv[n].alpha = alpha
                self.patterns[id].sv[n].weight = weight
            else:
                self._delete_pattern(id, n)
    
    def _delete_pattern(self, id, n):
        sv_num = len(self.patterns[id].sv)
        if sv_num < 2 and id > 0:
            del self.patterns[id]
        else:
            del self.patterns[id].sv[n]

    def _maintain_sv_budgets(self):
        while sum([len(self.patterns[p].sv) for p in self.patterns]) > sv_max:
            (p_id, sv_id, min_weight) = (0, 0, np.inf)
            for p in self.patterns:
                for i in self.patterns[p].sv:
                    if p == 0 and i == 0:
                        continue
                    weight = self.patterns[p].sv[i].weight
                    if weight < min_weight:
                        min_weight = weight
                        (p_id, sv_id) = (p, i)
            alpha = self.patterns[p_id].sv[sv_id].alpha
            self.w = self.w - alpha * self.patterns[p_id].dX[sv_id]
            if len(self.patterns[p_id].sv) < 2:
                del self.patterns[p_id]
            else:
                del self.patterns[p_id].sv[sv_id]
            

class Pattern:
    def __init__(self, Xi, Yi, X, Y):
        self.Xi = np.array(Xi)
        self.Yi = np.array(Yi)
        self.dX = np.array(X)
        self.Y = np.array(Y)
        self.sv = {}
        self.loss = self._compute_loss()
    
    # compute cost table
    def _compute_loss(self):
        (x, y, w, h) = self.Yi.reshape(4, 1).repeat(self.Y.shape[0], axis=1)
        (x1, y1, w1, h1) = self.Y.T
        (x1_, y1_, x_, y_) = (x1+w1, y1+h1, x+w, y+h)
        (left, top) = (np.maximum(x, x1), np.maximum(y, y1))
        (right, bottom) = (np.minimum(x_, x1_), np.minimum(y_, y1_))
        intersect = (bottom - top) * (right - left)
        intersect[intersect < 0] = 0
        union = w * h + w1 * h1 - intersect
        return 1 - intersect / union        

class SupportVector:
    def __init__(self, alpha=10e-8, weight=0):
        self.alpha = alpha
        self.weight = weight