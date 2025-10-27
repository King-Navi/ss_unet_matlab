clear; clc;

modelPath    = "unet_modern.mat";
imagePath    = fullfile("data","polypgen","img","C1_102OLCV1_100H0006.jpg");
outputMask   = "prediction_mask.png";
outputOverlay= "prediction_overlay.png";
classNames   = ["background","Polyp"];
targetSize   = [256 256];
polypColor   = [1 0 0];    % rojo overlay

S = load(modelPath);
net = S.net;

Iorig = imread(imagePath);
if size(Iorig,3)==1
    I3 = repmat(Iorig,1,1,3);
else
    I3 = Iorig;
end

% -- Asegura tamaño de entrada de la red --
resized = false;
if any(size(I3,1:2) ~= targetSize)
    Iin = imresize(I3, targetSize, "nearest");
    resized = true;
else
    Iin = I3;
end

% -------- Try semanticseg, fallback a predictor manual --------
try
    C = semanticseg(Iin, net, 'ExecutionEnvironment','auto');   % puede funcionar con dlnetwork en versiones recientes
    % Fuerza categorías y orden, por si semanticseg devuelve solo 1 clase:
    miss = setdiff(classNames, categories(C));
    if ~isempty(miss), C = addcats(C, miss); end
    C = reordercats(C, classNames);
catch
    % Predictor manual (dlnetwork): predict -> argmax -> categorical con mismas categorías
    X = dlarray(im2single(Iin), 'SSCB');   % HxWxCx1, single [0..1]
    Y = predict(net, X);                   % HxWxCx1 (probabilidades softmax)
    Y = extractdata(Y);
    [~,idx] = max(Y,[],3);                 % 1=bg, 2=Polyp
    C = categorical(idx-1, [0 1], classNames);  % mapea 1->0(bg), 2->1(Polyp)
end

% Si se reescalaste regresa la máscara al tamaño original --
if resized
    maskSmall = (C == classNames(2));
    mask = imresize(maskSmall, size(Iorig,[1 2]), 'nearest');
else
    mask = (C == classNames(2));
end

% Guardar máscara 0/255
imwrite(uint8(mask)*255, outputMask);

% Overlay visual
cmap = [0 0 0; polypColor];
Crow = categorical(mask,[0 1],classNames);  % asegurar cats para overlay
overlay = labeloverlay(Iorig, Crow, 'Colormap', cmap, 'Transparency', 0.6);
imwrite(overlay, outputOverlay);

% Mostrar
figure;
subplot(1,3,1); imshow(Iorig);  title("Original");
subplot(1,3,2); imshow(mask);    title("Predicted Mask");
subplot(1,3,3); imshow(overlay); title("Overlay");
