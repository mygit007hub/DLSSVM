%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	Implemetation of the tracker described in paper
%	"MEEM: Robust Tracking via Multiple Experts using Entropy Minimization",
%   Jianming Zhang, Shugao Ma, Stan Sclaroff, ECCV, 2014
%
%	Copyright (C) 2014 Jianming Zhang
%
%	This program is free software: you can redistribute it and/or modify
%	it under the terms of the GNU General Public License as published by
%	the Free Software Foundation, either version 3 of the License, or
%	(at your option) any later version.
%
%	This program is distributed in the hope that it will be useful,
%	but WITHOUT ANY WARRANTY; without even the implied warranty of
%	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%	GNU General Public License for more details.
%
%	You should have received a copy of the GNU General Public License
%	along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%	If you have problems about this software, please contact: jmzhang@bu.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function config = makeConfig(frame,selected_rect,use_color,use_experts,use_iif, show_img)
warning('off','MATLAB:maxNumCompThreads:Deprecated');
maxNumCompThreads(1); % The rounding error due to using different number of threads
                      % could cause different tracking results.  On most sequences,
                      % the difference is very small, while on some challenging
                      % sequences, the difference can be substantial due to
                      % "butterfly effects". Therefore, we suggest using
                      % Spatial Robustness Evaluation (SRE) to benchmark
                      % trackers.
if nargin < 5
    use_iif = true;
end

%% default setting up
config.search_roi = 2; % ratio of the search roi to tracking window
config.padding = 40; % for object out of border

config.debug = false;
config.verbose = false;
config.display = show_img; % show tracking result at runtime
% config.use_experts = use_experts;
config.use_raw_feat = false; % raw intensity feature value
config.use_iif = use_iif; % use illumination invariant feature

% config.svm_thresh = -0.7; % for detecting the tracking failure
% config.max_expert_sz = 4;
% config.expert_update_interval = 50;
% config.update_count_thresh = 1;
% config.entropy_score_winsize = 5;
% config.expert_lambda = 10;
% config.label_prior_sigma = 15;

config.hist_nbin = 32; % histogram bins for iif computation

config.thresh_p = 0.1; % IOU threshold for positive training samples
config.thresh_n = 0.5; % IOU threshold for negative ones


%  automatic setting up for determining grid sample step and feature channels
%  (do not change the following)

% check if the frame is in RGB format
config.use_color = false;
if (size(frame,3) == 3 && ~isequal(frame(:,:,1),frame(:,:,2),frame(:,:,3))) && use_color
    config.use_color = true;
end



% getFeatureRep will use these.
% decide feature channel number
if config.use_color
    thr_n = 9;
else
    thr_n = 17;
end
config.thr = (1/thr_n:1/thr_n:1-1/thr_n)*255;
% 1.config.use_color = true  ->  config.thr = [0.2 , 0.4, 0.6, 0.8]         * 255
% 2.config.use_color = flase ->  config.thr = [1/9, 2/9, 3/9, ... , 8/9]    * 255
config.fd = numel(config.thr);
% 1.    4
% 2.    8




% decide image scale and pixel step for sampling feature
% rescale raw input frames propoerly would save much computation
frame_min_width = 320;
trackwin_max_dimension = 64;

% How many templates to select from sampler.roi (training examples where y != yi)
template_max_numel = 400;

frame_sz = size(frame);
% For basketball :  432 x 576 x 3 


% selected_rect is the first selected tracking window: [x, y, width, height]


% *********************************

% ?????????? for what ????????????????

% **********************************


% if selected_rect's width or height(81) < trackwin_max_dimension(64) || frame's width(576) < frame_min_width(320) 
if max(selected_rect(3:4)) <= trackwin_max_dimension || frame_sz(2) <= frame_min_width
    config.image_scale = 1;
else
    % min_wid_scale = 320 / 576 = 0.5556        
    scaled_wid = frame_min_width/frame_sz(2);
    % min_rect = 64 / max( width, height ) = 0.7901
    scaled_rect = trackwin_max_dimension / max(selected_rect(3:4));
    config.image_scale = max(scaled_rect, scaled_wid);
end

wh_rescale = selected_rect(3:4) * config.image_scale;
win_area = prod(wh_rescale); % tracking window area

config.ratio = sqrt(template_max_numel / win_area);

template_sz = round(wh_rescale * config.ratio);
config.template_sz = template_sz([2 1]);
