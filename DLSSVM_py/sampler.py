import numpy as np
import skimage.color as color

class Sampler:
    def __init__(self, frame, rect, scale=1.5, step=1):
        region = self._get_region(rect, scale, frame)
        cropped = self._crop_region(frame, region)
        self.feat_region = self._get_features(cropped)
        self.region = region
        self.frame = frame
        self.rect = np.array(rect, dtype=int)
        self.scale = scale
        self.step = step
        self.samples = None

    def get_samples(self):
        if not self.samples:
            self._sample()
        return self.samples

    def crop(self, rect):
        (x, y, w, h) = rect
        (x0, y0) = self.rect[0:2]
        return self.feat_region[y-y0:y-y0+h, x-x0:x-x0+w].ravel()
    
    def _sample(self):
        (x0, y0) = self.region[0:2]
        (h0, w0) = self.feat_region.shape[0:2]
        (w, h) = self.rect[2:4]
        (X, Y) = ([], [])
        for j in range(0, h0-h, self.step):
            for i in range(0, w0-w, self.step):
                X += [self.feat_region[j:j+h, i:i+w].ravel()]
                Y += [[i+x0, j+y0, w, h]]
        self.samples = (np.array(X), np.array(Y))


    def _get_region(self, rect, scale, frame):
        (x, y, w, h) = tuple(rect)
        r = round(scale * (w + h) / 2)
        region = np.array([x+w/2-r, y+h/2-r, 2*r, 2*r], dtype=int)
        if region[0] < 0:
            region[0] = 0
        elif region[0] + region[2] > frame.shape[1]:
            region[0] = frame.shape[1] - region[2]
        if region[1] < 0:
            region[1] = 0
        elif region[1] + region[3] > frame.shape[0]:
            region[1] = frame.shape[0] - region[3]
        return region
    
    def _crop_region(self, frame, region):
        (x, y, w, h) = tuple(region)
        return frame[y:y+y, x:x+w]

    def _get_features(self, cropped):
        lab = color.rgb2lab(cropped)
        gray = color.rgb2gray(cropped)
        gray = gray.reshape(*gray.shape, 1)
        return np.concatenate((lab, gray), axis=2)