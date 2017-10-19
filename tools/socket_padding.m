% padding 2d image to fit the net
normImage     = false;
scaleImage    = 1;
nTiles        = 1;

tmp = h5read('socket/train_socket.h5', '/data');
rawstack = squeeze(tmp);
clear tmp;
% rawstack(rawstack(:)>255) = 255;
% rawstack(rawstack(:)<0) = 0;
%
%  rescale images if requested
%
if (normImage)
  rawstack = rawstack - min(rawstack(:));
  rawstack = rawstack / max(rawstack(:));
  for t=1:size(rawstack,3)
	slice = rawstack(:,:,t);
	slice = slice - median(slice(:));
	rawstack(:,:,t) = slice;
  end	
end

if (scaleImage ~= 1)
  rawstack = imresize( rawstack, scaleImage, 'bilinear');
end

%
%  Do the segmenation
%

data = reshape( single(rawstack), ...
				[size(rawstack,1), size(rawstack,2), 1 size(rawstack,3)]);
%testIdx = 1:5; %size(data,4);
%data = data(:,:,:,testIdx);

opts.n_tiles = nTiles;
%opts.padding = 'mirror';
opts.padding = 'zero';
opts.downsampleFactor = 16;
d4a_size= 0;
opts.padInput =   (((d4a_size *2 +2 +2)*2 +2 +2)*2 +2 +2)*2 +2 +2;
opts.padOutput = ((((d4a_size -2 -2)*2-2 -2)*2-2 -2)*2-2 -2)*2-2 -2;
opts.average_mirror = false;

%  compute input and output sizes (for v-shaped 4-resolutions network)
%
d4a_size = ceil(([size(data,1) ceil(size(data,2)/opts.n_tiles)] - opts.padOutput)/opts.downsampleFactor);
input_size = opts.downsampleFactor*d4a_size + opts.padInput;
output_size = opts.downsampleFactor*d4a_size + opts.padOutput;
%disp(['d4a_size = ' num2str(d4a_size) ' --> insize = ' num2str(input_size) ...
%      ', outsize = ' num2str(output_size)])

%
%  create padded volume mit maximal border
%
border = round(input_size-output_size)/2; % nearest integer
paddedFullVolume = zeros(size(data,1) + 2*border(1), ...
                         size(data,2) + 2*border(2), ...
                         size(data,3),...
                         size(data,4), 'single');

paddedFullVolume( border(1)+1:border(1)+size(data,1), ...
                  border(2)+1:border(2)+size(data,2), ...
                  :, : ) =data;


if( strcmp(opts.padding,'mirror'))
  xpad  = border(1);
  xfrom = border(1)+1;
  xto   = border(1)+size(data,1);
  paddedFullVolume(1:xfrom-1,:,:) = paddedFullVolume( xfrom+xpad:-1:xfrom+1,:,:);
  paddedFullVolume(xto+1:end,:,:) = paddedFullVolume( xto-1:-1:xto-xpad,    :,:);
  
  ypad  = border(2);
  yfrom = border(2)+1;
  yto   = border(2)+size(data,2);
  paddedFullVolume(:, 1:yfrom-1,:) = paddedFullVolume( :, yfrom+ypad:-1:yfrom+1,:);
  paddedFullVolume(:, yto+1:end,:) = paddedFullVolume( :, yto-1:-1:yto-ypad,    :);
end

%
%  do the classification (tiled)
%  average over flipped images
%data_padding = zeros([284 380 1 1280]);
%data_mirror = zeros([284 380 1 1280]);
%data_padding = zeros([380 560 1 7631]); %(7631, 1, 367, 192)
%data_mirror = zeros([380 560 1 7631]);
data_padding = zeros([input_size size(data,3) size(data,4)]);
for num=1:size(data,4)
  disp(['segmenting image ' num2str(num)])
  tic
  % crop input data
  for yi=0:opts.n_tiles-1
    paddedInputSlice = zeros([input_size size(data,3)], 'single');
    validReg(1) = min(input_size(1), size(paddedFullVolume,1));
    validReg(2) = min(input_size(2), size(paddedFullVolume,2) - yi*output_size(2));
    paddedInputSlice(1:validReg(1), 1:validReg(2),:) = ...
		paddedFullVolume(1:validReg(1), yi*output_size(2)+1:yi*output_size(2)+validReg(2), :, num);
  
%     scores_caffe = caffe('forward', {paddedInputSlice});
    data_padding(:,:,1,num) = paddedInputSlice;
	
% 	if( opts.average_mirror == true)
%       data_mirror(:,:,1,num) = fliplr(paddedInputSlice);
% % 	  scores_caffe = caffe('forward', {fliplr(paddedInputSlice)});
%       scores_caffe = net.forward({fliplr(paddedInputSlice)});
% 	  scoreSlice = scoreSlice+fliplr(scores_caffe{1});
%       scores_caffe = net.forward({flipud(paddedInputSlice)});
% 	  scoreSlice = scoreSlice+flipud(scores_caffe{1});
%       scores_caffe = net.forward({flipud(fliplr(paddedInputSlice))});
% 	  scoreSlice = scoreSlice+flipud(fliplr(scores_caffe{1}));
% 	  scoreSlice = scoreSlice/4;
% 	end

% 	if( num==1 && yi==0)
% 	  nClasses = size(scoreSlice,3);
% 	  scores = zeros( size(data,1), size(data,2), nClasses, size(data,4));
% 	end
% 	%    figure(4); imshow( reshape(scores, [size(scores,1) size(scores,2)*size(scores,3)]),[])
%     validReg(1) = min(output_size(1), size(scores,1));
%     validReg(2) = min(output_size(2), size(scores,2) - yi*output_size(2));
%     scores(1:validReg, yi*output_size(2)+1:yi*output_size(2)+validReg(2),:,num) = ...
% 		scoreSlice(1:validReg(1),1:validReg(2),:);
  end
  toc
end

label = h5read('socket/train_socket.h5', '/label');
hdf5write( 'socket/train_p_socket.h5', ...
		   'data', data_padding, ...
		   'labels', label);
disp('Done!')
% caffe('reset')
% caffe.reset_all();
