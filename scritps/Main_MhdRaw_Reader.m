clear all
close all
file='Case48';
% Assuming your MHD and RAW files are in the same directory
mhdFilePath = strcat(file,'_segmentation.mhd'); 

[headerInfo, rawFileName] = readMHDHeader(mhdFilePath);
rawFilePath = fullfile(fileparts(mhdFilePath), rawFileName); 
Mask_volume = readRAWVolume_mhd(rawFilePath, headerInfo);

rawFilePath=strcat(file,'.raw');
Data_volume = readRAWVolume_raw(rawFilePath, headerInfo);

% Now 'volume' contains your 3D image data. You can visualize it:
% sliceViewer(volume); % Requires Image Processing Toolbox
% or
% imshow(volume(:,:,sliceNumber), []); % To view a single slice