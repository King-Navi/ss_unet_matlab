function train_unet_modern
% unet (dlnetwork) + trainnet + loss (Weighted CE + Soft Dice)
% imgs 256x256 (RGB or grises) y binary masks (Polyp vs background).

%libera toda la memoria de la GPU, cierra el contexto CUDA
reset(gpuDevice);

imDir   = fullfile("data","polypgen","img");
maskDir = fullfile("data","polypgen","mask");
classNames = ["background","Polyp"];   % mantener orden 
labelIDs   = [0 255];

assert(isfolder(imDir),  "Images folder not found: %s", imDir);
assert(isfolder(maskDir),"Masks folder not found: %s", maskDir);

%% Datastores (Computer Vision Toolbox) igual que dataloaders
imds = imageDatastore(imDir, ...
    'IncludeSubfolders', true, ...
    'FileExtensions', {'.png','.jpg','.jpeg','.tif','.tiff'});

% ReadFcn sea un handle que reciba un solo argumento (el filename) y devuelva una matriz categórica
pxds = pixelLabelDatastore(maskDir, classNames, labelIDs, ...
    'IncludeSubfolders', true, ...
    'FileExtensions', {'.png','.jpg','.jpeg','.tif','.tiff'}, ...
    'ReadFcn', @(fn) readMaskAsCategorical(fn, classNames));

%% verificar pares
[imds, pxds, rep] = alignByBasename(imds, pxds, false);
fprintf('Kept pairs: %d | Missing masks: %d | Extra masks: %d\n', ...
    rep.keptPairs, rep.missingMasks, rep.extraMasks);
if rep.missingMasks > 0
    disp('Examples without matching mask:'); disp(string(rep.sampleMissing(:)));
end

totalImages = numel(imds.Files);
assert(totalImages > 1, "Not enough paired samples.");

%% split: ensure validation has positive cases
% Semilla fija (reproducible)
rng(123);
% F = false(___,like=p) devuelve un arreglo de ceros lógicos de la misma dispersión que la variable lógica p
hasPolyp = false(totalImages,1);
%Detectar qué imágenes tienen pólipo
for i = 1:totalImages
    % reads the Ith image file from the datastore 
    L = readimage(pxds, i);
    % matriz categórica (H×W) con categorías ["background","Polyp"]. L(:) convierte toda la matriz en un vector columna.
    hasPolyp(i) = any(L(:) == "Polyp");
end

% find te devuelve los índices
posIdx = find(hasPolyp);
% ~ es ! osea NOT
negIdx = find(~hasPolyp);

valFrac   = 0.30;
% numel devuelve el número de elementos
numValPos = max(1, round(valFrac * numel(posIdx)));
numValNeg = max(1, round(valFrac * numel(negIdx)));

valIdx = [randsample(posIdx, min(numValPos, numel(posIdx)), false); ...
          randsample(negIdx, min(numValNeg, numel(negIdx)), false)];
valIdx = unique(valIdx);
trnIdx = setdiff(1:totalImages, valIdx);

imdsTrain = subset(imds, trnIdx);
pxdsTrain = subset(pxds, trnIdx);
imdsVal   = subset(imds, valIdx);
pxdsVal   = subset(pxds, valIdx);

%% Patch-based training

%une datastores por indice, asegira con preprocessTrainPair que si sea 3
%canales tipo single
dsTrainNN = transform(combine(imdsTrain, pxdsTrain), @(d) preprocessTrainPair(d, classNames));

%% Build U-Net (modern API -> dlnetwork)
imageSize    = [256 256 3];
numClasses   = numel(classNames);
encoderDepth = 4;


net0 = unet(imageSize, numClasses, 'EncoderDepth', encoderDepth); % dlnetwork

%% Training options
optNN = trainingOptions("adam", ...
    "InitialLearnRate", 1e-4, ...
    "MaxEpochs", 220, ...
    "MiniBatchSize", 8, ...
    "ExecutionEnvironment", "auto", ...
    "VerboseFrequency", 1000);

%% Loss function: Weighted CE + Soft Dice (focus on Polyp)
classWeights = [1 20];
lambdaDice   = 1.0;

lossFcn = @(Y,T) lossDiceCE(Y,T,classNames,classWeights,lambdaDice);

%% Train
net = trainnet(dsTrainNN, net0, lossFcn, optNN);

%% Save
save("unet_modern.mat","net");



%% Evaluate (MATLAB metrics)
try
    pxdsPred = predictSegDL(imdsVal, net, classNames);
    metrics  = evaluateSemanticSegmentation(pxdsPred, pxdsVal, 'Verbose', false);
    disp('Class metrics (MATLAB):'); disp(metrics.ClassMetrics);

    [polyIoU, polyDice, polyPrec, polyRec] = polypOnlyMetrics(pxdsVal, pxdsPred);
    fprintf('Polyp IoU:  %.4f\n', polyIoU);
    fprintf('Polyp Dice: %.4f\n', polyDice);
    fprintf('Polyp Prec: %.4f | Polyp Rec: %.4f\n', polyPrec, polyRec);
catch ME
    warning("Evaluation failed: %s", string(ME.message));
end

%% Polyp-only, background-agnostic metrics (more realistic)
[polyIoU, polyDice, polyPrec, polyRec, ~] = polypOnlyMetrics(pxdsVal, pxdsPred);
fprintf('Polyp IoU:  %.4f\n', polyIoU);
fprintf('Polyp Dice: %.4f\n', polyDice);
fprintf('Polyp Prec: %.4f | Polyp Rec: %.4f\n', polyPrec, polyRec);

%% Quick visual check
%I1 = readimage(imdsVal, 1);
%L1 = readimage(pxdsVal, 1);
%P1 = readimage(pxdsPred, 1);
%figure;
%subplot(1,3,1); imshow(I1); title('Image');
%subplot(1,3,2); imshow(labeloverlay(I1, L1=='Polyp','Transparency',0.6)); title('GT overlay');
%subplot(1,3,3); imshow(labeloverlay(I1, P1=='Polyp','Transparency',0.6)); title('Pred overlay');

end

%%

function C = readMaskAsCategorical(filename, classNames)
    % Read mask and convert to categorical {background, Polyp}
    % Si gris [HxW] (RGB: SI RGB es H×W×3.)
    M = imread(filename);
    
    %fuerza si mayor a 128  entonces 1
    if size(M,3) > 1, M = rgb2gray(M); end
    M = M >= 128;  % threshold to 0/1
    % categorical (Retorna un arreglo categórico del mismo tamaño que la entrada)
    C = categorical(M, [0 1], classNames);
end

% imdsIn Lista de rutas de imágenes (imageDatastore)
% pxdsIn Lista de rutas de "mascaras" (pixelLabelDatastore)
function [imdsOut, pxdsOut, report] = alignByBasename(imdsIn, pxdsIn, strict)
    % Align images and masks by normalized basename.

    % nargin devuelve el número de argumentos de entrada de una función que se proporcionan al llamar a la función
    if nargin < 3, strict = false; end
    
    imFiles = imdsIn.Files;
    pxFiles = pxdsIn.Files;
    
    % cellfun: aplica una función a cada elemento de un cell array y devuelve los resultados.
    % @normKey function handle
    imKeys = cellfun(@normKey, imFiles, 'UniformOutput', false);
    pxKeys = cellfun(@normKey, pxFiles, 'UniformOutput', false);
    
    %C = unique(A) devuelve los mismos datos que en A, pero sin repeticiones.
    
    % pxKeys es el cellfun
    % resultado 1 Datos únicos de A
    % resultado 2 inidces de resultado 1devuelto como un vector columna de índices a la primera aparición de elementos repetidos
    % resultado 3 arreglo del mismo tamaño que A donde cada elemento indica en qué posición de C
    [keysU, ~, idxU] = unique(pxKeys);
    
    firstIdx = accumarray(idxU(:), (1:numel(pxFiles))', [], @(v) v(1));
    maskMap = containers.Map(keysU, num2cell(firstIdx));
    
    matchedImIdx = [];
    matchedPxIdx = [];
    missingList  = {};
    
    for i = 1:numel(imFiles)
        k = imKeys{i};
        if isKey(maskMap, k)
            % end+1 es append
            matchedImIdx(end+1) = i; %#ok<AGROW>
            % maskMap(k) devuelve el índice de la máscara que corresponde a la clave k
            matchedPxIdx(end+1) = maskMap(k); %#ok<AGROW>
        else
            missingList{end+1} = imFiles{i}; %#ok<AGROW>
        end
    end
    
    extraMasksIdx = setdiff(1:numel(pxFiles), matchedPxIdx);
    
    if strict && ~isempty(missingList)
        exampleList = strjoin(missingList(1:min(10, end)), newline);
        error('alignByBasename:MissingMasks', ...
            'Images without matching mask: %d\nExamples:\n%s', ...
            numel(missingList), exampleList);
    end
    
    if isempty(matchedImIdx)
        error('alignByBasename:NoPairs', 'No (image, mask) pairs found.');
    end
    
    imdsOut = subset(imdsIn, matchedImIdx);
    pxdsOut = pixelLabelDatastore(pxFiles(matchedPxIdx), pxdsIn.ClassNames, [0 255]);
    pxdsOut.ReadFcn = pxdsIn.ReadFcn;
    
    report = struct( ...
        'totalImages',   numel(imFiles), ...
        'totalMasks',    numel(pxFiles), ...
        'keptPairs',     numel(matchedImIdx), ...
        'missingMasks',  numel(missingList), ...
        'extraMasks',    numel(extraMasksIdx), ...
        'sampleMissing', {missingList(1:min(10,end))}, ...
        'sampleExtra',   {pxFiles(extraMasksIdx(1:min(10,end)))} ...
    );
end

function k = normKey(pathStr)
    % extrae el basename sin extensión,
    [~, name, ~] = fileparts(pathStr);
    name = lower(name);
    name = regexprep(name, '(_|-|\s)(mask|segmentation|gt|label|lbl)$', '', 'once');
    name = regexprep(name, '[^a-z0-9]+', '');
    k = name;
end

% predicciones sobre tu conjunto de validación y te devuelve un pixelLabelDatastore con las máscaras predichas
function pxdsPred = predictSegDL(imdsVal, net, classNames)
    % Manual predictor for dlnetwork -> pixelLabelDatastore of predictions.
    outDir = tempname; mkdir(outDir);
    predFiles = strings(numel(imdsVal.Files),1);
    
    for i=1:numel(imdsVal.Files)
        I = readimage(imdsVal,i);
        % garantiza 3 canales
        if size(I,3)==1, I = repmat(I,1,1,3); end
        X = dlarray(single(I)/255, 'SSCB'); % escala a [0,1] y etiqueta dims (S,S,C,B) Spatial, Spatial, Channel, Batch (aquí batch N=1).
        Y = predict(net, X); % salida: probabilidades (softmax) HxWxCx1
        Y = extractdata(Y); % dlarray/GPU a arreglo normal
        [~,idx] = max(Y, [], 3); % argmax sobre la dimensión de clases
        L = categorical(idx-1, [0 1], classNames); % mapea 0 bg, 1 Polyp (orden [bg,Polyp])
        % Escribe a disco
        imwrite(uint8(L=="Polyp")*255, fullfile(outDir, sprintf('pred_%05d.png',i)));
        predFiles(i) = fullfile(outDir, sprintf('pred_%05d.png',i));
    end
    
    pxdsPred = pixelLabelDatastore(predFiles, classNames, [0 255], ...
        'ReadFcn', @(fn) readMaskAsCategorical(fn, classNames));
end


% Y: HxWxCxN probabilidades (salida softmax de la U-Net)
% T: HxWx1xN máscara objetivo numérica 0/1 (1=Polyp), es dlarray en trainnet
% lambdaDice equilibrar cuánto influye Dice frente a la Cross-Entropy (CE) por píxel.
function loss = lossDiceCE(Y, T, classNames, classWeights, lambdaDice)

    eps = 1e-7;
    % eviat 0/1 ya que -log(0)
    Y = clampProb(Y, eps);
    
    % la matriz numérica (numeric dlarray) sea única para la máscara de pólipos de destino
    tPolyp = single(T);% HxWx1xN, dlarray
    bgMask = 1 - tPolyp;% background mask
    
    % cat concatena  arreglos a lo largo de una dimensión específica.
    % bgMask es 1 - tPolyp, matriz H×W×1×N con 1 donde es fondo y 0 donde hay pólipo.
    % tPolyp es la máscara H×W×1×N con 1 donde hay pólipo y 0 en fondo
    OH = cat(3, bgMask, tPolyp);
    % OH(:,:,1,:) = bgMask donde 1 si el píxel es background, 0 si no.
    % OH(:,:,2,:) = tPolyp donde 1 si el píxel es Polyp, 0 si no.
    
    % Weighted Cross-Entropy
    Wbg = classWeights(1);
    Wpo = classWeights(2);
    % mapa de pesos por canal
    Wmap = cat(3, bgMask*0 + Wbg, tPolyp*0 + Wpo);
    
    % NOTA: Para calcular la CE, solo importa la probabilidad que el modelo asigna a la clase verdadera; las demás clases no contribuyen (tienen 0 )
    CE = -OH .* log(Y);
    CE = CE .* Wmap;
    % antes es H×W×C×N (píxel × píxel × clase × batch)
    CE = mean(CE,'all');
    
    % Soft Dice on Polyp channel
    % prob. predicha de Polyp (canal 2 de Y: HxWx1xN)
    y = Y(:,:,2,:);% predicted prob for Polyp
    t = tPolyp;        % GT Polyp mask 0/1
    inter = sum(y.*t,'all'); % intersección suave
    card  = sum(y,'all') + sum(t,'all'); % |y| + |t|
    dice  = (2*inter + 1) / (card + 1); % suavizado = 1 para evitar 0/0
    diceLoss = 1 - dice; % pérdida de Dice
    
    loss = CE + lambdaDice * diceLoss;
end


function Y = clampProb(Y, eps)
Y = min(max(Y, eps), 1-eps);
end

% parametros ambos datastores
function [miou, dice, prec, rec, cm] = polypOnlyMetrics(pxdsGT, pxdsPR)
    % Ignora backgrouind
    % Accumulators across all images (flattened pixel-wise)
    % ground truth all píxeles de la verdad de terreno (GT) aplanados y concatenados.
    % prediction all píxeles de la predicción (PR) aplanados y concatenados.
    gtAll = []; prAll = [];
    numImages = numel(pxdsGT.Files);
    for i=1:numImages
        GT = readimage(pxdsGT, i) == "Polyp";
        PR = readimage(pxdsPR, i) == "Polyp";
        gtAll = [gtAll; GT(:)]; %#ok<AGROW>
        prAll = [prAll; PR(:)]; %#ok<AGROW>
    end
    % gtAll(p) = 1 si el píxel p es Polyp en la verdad (GT), si no 0
    % prAll(p) = 1 si el píxel p fue predicho como Polyp, si no 0.
    % Confusion components for the Polyp class
    TP = sum(gtAll & prAll);
    FP = sum(~gtAll & prAll);
    FN = sum(gtAll & ~prAll);
    % IoU (Jaccard) for Polyp
    denIoU = TP + FP + FN;
    miou = (denIoU>0) * (TP / max(denIoU,1));

    % Dice for Polyp
    dice = (TP>0 || (FP+FN)>0) * (2*TP / max(2*TP + FP + FN, 1));
    
    % Precision & Recall for Polyp
    prec = TP / max(TP+FP, 1);
    rec  = TP / max(TP+FN, 1);
    cm = [TP FP; FN 0]; %#ok<NASGU>
end


% toma un par {imagen, máscara} y convierte a 3 canales tipo single valores [0,1].
function out = preprocessTrainPair(in, classNames)
    % Ensures 3-channel images (single [0,1]) and numeric 0/1 label (Polyp mask)
    I = in{1};
    L = in{2};
    
    % Si I es gris (C=1), la replica a 3 canales (H×W×3)
    if size(I,3)==1
        I = repmat(I,1,1,3);
    end
    I = im2single(I);
    
    % Produce numeric binary mask: 1=Polyp, 0=background
    if iscategorical(L)
        M = (L == classNames(2));% logical
    else
        if ndims(L)==3 && size(L,3)>1, L = rgb2gray(L); end
        M = L >= 128;% logical
    end
    
    L = single(M);% single 0/1, compatible with dlarray
    out = {I, L};
end

