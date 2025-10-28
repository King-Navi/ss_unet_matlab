clear; clc;

modelPath    = "unet_modern.mat";
imagePath    = fullfile("data","polypgen","img","C1_102OLCV1_100H0006.jpg");
gtMaskPath = fullfile("data","polypgen","mask","C1_102OLCV1_100H0006_mask.png")
outputMask   = "prediction_mask.png";
outputOverlay= "prediction_overlay.png";
classNames   = ["background","Polyp"];
targetSize   = [256 256];
polypColor   = [1 0 0];    % rojo overlay

S = load(modelPath);
net = S.net;

% Canales 3 asegurarse
Iorig = imread(imagePath);
if size(Iorig,3)==1
    I3 = repmat(Iorig,1,1,3);
else
    I3 = Iorig;
end

% reescalaste
resized = false;
if any(size(I3,1:2) ~= targetSize)
    Iin = imresize(I3, targetSize, "nearest");
    resized = true;
else
    Iin = I3;
end


try
    % puede funcionar con dlnetwork en versiones recientes

    % 'ExecutionEnvironment','auto': usa GPU si hay disponible; si no, CPU.
    C = semanticseg(Iin, net, 'ExecutionEnvironment','auto');   
    % Fuerza categorías y orden, por si semanticseg devuelve solo 1 clase:
    % categories(C): lista de categorías presentes en el categórico resultante C
    miss = setdiff(classNames, categories(C));
    % addcats: agrega las categorías faltantes a C para que siempre tenga las mismas categorías globales (aunque alguna no aparezca en esa imagen).
    if ~isempty(miss), C = addcats(C, miss); end
    % reordercats: reordena las categorías de C para que queden exactamente en el orden classNames
    C = reordercats(C, classNames);
catch
    % Predictor manual (dlnetwork): predict -> argmax -> categorical con mismas categorías
    X = dlarray(im2single(Iin), 'SSCB');% HxWxCx1, single [0..1]
    Y = predict(net, X);% HxWxCx1 (probabilidades softmax)
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


%%
hasGT = ~isempty(gtMaskPath) && isfile(gtMaskPath);

P = logical(mask);
if hasGT
    GT = imread(gtMaskPath);
    if size(GT,3) > 1, GT = rgb2gray(GT); end
    % Robust binarization (works for 0/255 or JPEG-ish)
    if islogical(GT)
        GT = GT;
    elseif isa(GT,'uint8') || isa(GT,'uint16')
        GT = GT >= double(intmax(class(GT)))/2;
    else
        GT = GT >= 0.5; % if float in [0..1]
    end
    % Resize GT to predicted mask size if needed
    if any(size(GT) ~= size(P))
        GT = imresize(GT, size(P), 'nearest');
    end

    % Confusion components
    TP = sum(P(:)  &  GT(:));
    FP = sum(P(:)  & ~GT(:));
    FN = sum(~P(:) &  GT(:));
    TN = sum(~P(:) & ~GT(:));

    % Metrics (Polyp-only)
    IoU  = TP / max(TP + FP + FN, 1);
    Dice = 2*TP / max(2*TP + FP + FN, 1);
    Prec = TP / max(TP + FP, 1);
    Rec  = TP / max(TP + FN, 1);
    Acc  = (TP + TN) / numel(P);
    
    % '%.4f' = número en punto flotante con 4 decimales; '\n' = salto de línea.
    
    fprintf('Polyp metrics — IoU: %.4f | Dice: %.4f | Prec: %.4f | Rec: %.4f | Acc: %.4f\n', ...
        IoU, Dice, Prec, Rec, Acc);
    % IoU  (Intersection over Union, Jaccard): TP / (TP + FP + FN)
    % Proporción de solapamiento entre la predicción y el GT para la clase Polyp.
    %
    % Dice (F1 de segmentación): 2*TP / (2*TP + FP + FN)
    % Similar a IoU pero más sensible a regiones pequeñas; mide solapamiento.
    %
    % Prec (Precision, PPV): TP / (TP + FP)
    % De todo lo que predijiste como Polyp, ¿qué fracción era realmente Polyp?
    %
    % Rec  (Recall, Sensitivity, TPR): TP / (TP + FN)
    % De todo el Polyp real en GT, ¿qué fracción detectaste?
    %
    % Acc  (Accuracy global): (TP + TN) / (TP + TN + FP + FN)
    % Exactitud total. NOTA: puede ser engañosa si el fondo domina (clase desbalanceada).
    %
    % Todas están en [0,1]; mayor es mejor.


    % Difference overlay (TP=green, FP=red, FN=blue)
    tpMask =  P &  GT;
    fpMask =  P & ~GT;
    fnMask = ~P &  GT;
    labelMap = uint8(tpMask)*1 + uint8(fpMask)*2 + uint8(fnMask)*3; % 0:bg,1:TP,2:FP,3:FN
    diffOverlay = labeloverlay(Iorig, labelMap, ...
        'Colormap', [0 0 0; 0 1 0; 1 0 0; 0 0 1], 'Transparency', 0.55);
    imwrite(diffOverlay, "prediction_diff_overlay.png");

    % figure pane
    figure;
    % subplot(2,2,2): crea/selecciona el panel #2 de una grilla de 2 filas × 2 columnas (orden por filas: 1=arriba-izq, 2=arriba-der, 3=abajo-izq, 4=abajo-der).
    subplot(2,2,1); imshow(Iorig); title('Original');
    subplot(2,2,2); imshow(GT);    title('GT Mask');
    subplot(2,2,3); imshow(P);     title('Predicted Mask');
    subplot(2,2,4); imshow(diffOverlay); title('Diff: TP(g) FP(r) FN(b)');
else
    % No GT available
    pix = numel(P);
    polypPix = nnz(P);
    areaFrac = polypPix / pix;
    fprintf('No GT provided. Predicted polyp pixels: %d (%.4f of image)\n', polypPix, areaFrac);
end
