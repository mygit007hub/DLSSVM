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

function [feat F] = getFeatureRep(I,nbin)
% compute feature representation: mxnxd, d is the feature dimension
% decay factor and nbin is for the local histogram computation
%
global config

if size(I,3) == 3 && config.use_color
	I = uint8(255*RGB2Lab(I));
elseif size(I,3) == 3
	I = rgb2gray(I);
end

fd = config.fd;
% config.use_color == true - > fd = 4
% config.use_color == false - > fd = 8



% ?????????????????????????????????????????????
ksize = (1/config.ratio)*4;
if mod(ksize,2) == 0
	ksize = ksize + 1;
end
if config.use_iif
	F{1} = 255-calcIIF(I(:,:,1),[ksize ksize],nbin);
	% IIF2(I(:,:,1)*255, hist_mtx1, nbin);
	% feature by pixel ordering
% ??????????????????????????????????????????????? 

else
	F{1} = uint8(zeros([size(I,1),size(I,2)]));
end
F{2} = I(:,:,1); % gray image
if config.use_color
	F{3} = I(:,:,2); % color part
	F{4} = I(:,:,3); % color part
end

% Make a feature 
% F{1}   = IIF
% F{2:4} = Lab  ||  [No Color] F{2} = gray

if config.use_raw_feat
	feat = double(reshape(cell2mat(F),size(F{1},1),size(F{1},2),[]))/255;
	% change  F into matrix then divide by 255
else
	if ~config.use_color
		feat = zeros([size(I(:,:,1)),2*config.fd]);
	else
		feat = zeros([size(I(:,:,1)),4*config.fd]);
	end
	for i = 1:numel(F)
% if config.use_color
%     thr_n = 5;
% else
%     thr_n = 9;
% end
% config.thr = (1/thr_n:1/thr_n:1-1/thr_n)*255;
% % 1.config.use_color = true  ->  config.thr = [0.2 , 0.4, 0.6, 0.8]         * 255
% % 2.config.use_color = flase ->  config.thr = [1/9, 2/9, 3/9, ... , 8/9]    * 255
% config.fd = numel(config.thr);
% % 1.    4
% % 2.    8

		% for Every dimension of F :
		% 	F{i} copy config.fd(4) times, and then compare with config.thr each value [0.2*255, 0.4*255, 0.6*255, 0.8*255]
		% 	and then, stored it into feat (create for all zeros)
		feat(:,:,(i-1)*fd+1:i*fd) = bsxfun(@gt,repmat(F{i},[1 1 fd]), reshape(config.thr,1,1,[]));
		% size(feat) = ( size(I_crop) 16 )
	end
end

F = double(reshape(cell2mat(F),size(F{1},1),size(F{1},2),[]));
