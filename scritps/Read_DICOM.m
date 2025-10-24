%https://la.mathworks.com/help/images/ref/dicomread.html
 
clear all
close all

if 0
    %Para imágenes indexadas
    [X,map] = dicomread("Mask.dcm");
    n=31;
else
    info = dicominfo("Image.dcm");
    X = dicomread(info);
    n=1;
end

imshow(squeeze(X(:,:,1,n)),[]);