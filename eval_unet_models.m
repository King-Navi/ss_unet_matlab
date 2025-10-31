%usage

% results = eval_unet_models( ...
%    "unet_modern_2.mat", ...
%    "data/polypgen/img", ...
%    "data/polypgen/mask", ...
%    "split_idx.mat", ...   % o "" si no guardaste el split
%    [256 256 3], ...
%    345 ...
%);


% results = eval_unet_models("unet_modern.mat", "data/polypgen/img", "data/polypgen/mask", "", [256 256 3], 123);
% results = eval_unet_models("unet_modern_2.mat", "data/polypgen/img", "data/polypgen/mask", "", [256 256 3], 345);

function resultsTbl = eval_unet_models(modelPaths, imDir, maskDir, splitIdxPath, imageSize, seed)
% Evaluate one or more U-Net models on the 30% validation split.
% - modelPaths : string/cellstr of .mat files (must contain variable 'net' or 'dlnet')
% - imDir, maskDir : dataset folders
% - splitIdxPath : 'split_idx.mat' with trnIdx/valIdx; pass "" to recompute with seed
% - imageSize : e.g., [256 256 3]
% - seed : RNG seed used in training (e.g., 345)
%
% Returns:
%   resultsTbl: table with MATLAB and polyp-only metrics per model.

arguments
    modelPaths {mustBeText}
    imDir (1,1) string
    maskDir (1,1) string
    splitIdxPath (1,1) string = ""
    imageSize (1,3) double = [256 256 3]
    seed (1,1) double = 345
end
namefilecsvoutput = 'eval_results.csv';
assert(isfolder(imDir),  "Images folder not found: %s", imDir);
assert(isfolder(maskDir),"Masks folder not found: %s", maskDir);

classNames = ["background","Polyp"];
labelIDs   = [0 255];

% Build datastores
imds = imageDatastore(imDir, ...
    'IncludeSubfolders', true, ...
    'FileExtensions', {'.png','.jpg','.jpeg','.tif','.tiff'});

pxds = pixelLabelDatastore(maskDir, classNames, labelIDs, ...
    'IncludeSubfolders', true, ...
    'FileExtensions', {'.png','.jpg','.jpeg','.tif','.tiff'}, ...
    'ReadFcn', @(fn) readMaskAsCategorical(fn, classNames));

% Align pairs
[imds, pxds, rep] = alignByBasename(imds, pxds, false, labelIDs); %#ok<ASGLU>

totalImages = numel(imds.Files);
assert(totalImages > 1, "Not enough paired samples.");
fprintf("total de imagenes %d \n", totalImages);
% Obtain split
if splitIdxPath ~= "" && isfile(splitIdxPath)
    S = load(splitIdxPath);
    assert(isfield(S,'valIdx') && isfield(S,'trnIdx'), 'split_idx.mat must contain valIdx and trnIdx');
    valIdx = S.valIdx; trnIdx = S.trnIdx;
else
    % Recompute split deterministically (same logic as training)
    rng(seed);
    hasPolyp = false(totalImages,1);
    for i = 1:totalImages
        L = readimage(pxds, i);
        hasPolyp(i) = any(L(:) == "Polyp");
    end
    posIdx = find(hasPolyp);
    negIdx = find(~hasPolyp);
    valFrac   = 0.30;
    numValPos = max(1, round(valFrac * numel(posIdx)));
    numValNeg = max(1, round(valFrac * numel(negIdx)));
    valIdx = [randsample(posIdx, min(numValPos, numel(posIdx)), false); ...
              randsample(negIdx, min(numValNeg, numel(negIdx)), false)];
    valIdx = unique(valIdx);
    trnIdx = setdiff(1:totalImages, valIdx); %#ok<NASGU>
end

imdsVal = subset(imds, valIdx);
pxdsVal = subset(pxds, valIdx);
assert(numel(imdsVal.Files) == numel(pxdsVal.Files), 'imdsVal and pxdsVal length mismatch');

% Prepare results accumulator
modelPaths = cellstr(string(modelPaths));
rows = [];

for m = 1:numel(modelPaths)
    modelFile = modelPaths{m};
    assert(isfile(modelFile), "Model file not found: %s", modelFile);

    % Load network variable (net or dlnet)
    S = load(modelFile);
    if isfield(S,'net')
        net = S.net;
        %Calcular parametros 
        learnables = net.Learnables;
        totalParams = sum(cellfun(@(x) numel(x), learnables.Value));
        fprintf('Total learnable parameters: %d\n', totalParams);
    elseif isfield(S,'dlnet')
        net = S.dlnet;
    else
        error('Model %s does not contain variable ''net'' or ''dlnet''.', modelFile);
    end
    
    % ---- PREDICT & ACCUMULATE (no I/O) ----
    TP=0; FP=0; FN=0; TN=0;
    try
        Nval = numel(imdsVal.Files);
        pb = makeProgressBar(Nval, true, sprintf('Predicting: %s', string(modelFile)));
        cleaner = onCleanup(@() pb.close());
        useGPU = (exist('gpuDeviceCount','file') && gpuDeviceCount > 0);
    
        for i = 1:Nval
            % Read image + predict in-memory
            I0 = readimage(imdsVal, i);
            PR = predictMaskLogical(I0, net, imageSize, useGPU);  % logical mask (Polyp=true)
    
            % Read GT (categorical), align sizes, accumulate counts
            GTcat = readimage(pxdsVal, i);        % categorical with ["background","Polyp"]
            GT = (GTcat == "Polyp");
            if any(size(GT) ~= size(PR))
                PR = imresize(PR, size(GT), 'nearest');
            end
    
            TP = TP + sum(GT & PR,  'all');
            FP = FP + sum(~GT & PR, 'all');
            FN = FN + sum(GT & ~PR, 'all');
            TN = TN + sum(~GT & ~PR,'all');
    
            pb.update(i, Nval);
        end
        clear cleaner
    catch ME
        warning("Prediction failed for %s: %s", modelFile, ME.message);
    end
    
    % --- METRICS from counts (no files) ---
    [iou, dice, prec, rec, acc] = counts2metrics(TP,FP,FN,TN);
    
    rows = [rows; { string(modelFile), iou, dice, prec, rec, acc }];
    fprintf('Polyp metrics — IoU: %.4f | Dice: %.4f | Prec: %.4f | Rec: %.4f | Acc: %.4f\n', ...
        iou, dice, prec, rec, acc);
end

resultsTbl = cell2table(rows, 'VariableNames', ...
    {'Model','IoU','Dice','Precision','Recall','Accuracy'});

    try
        writetable(resultsTbl, namefilecsvoutput);
    catch
    end
end


function C = readMaskAsCategorical(filename, classNames)
    M = imread(filename);
    if size(M,3) > 1, M = rgb2gray(M); end
    M = M >= 128;
    C = categorical(M, [0 1], classNames);
end

function [imdsOut, pxdsOut, report] = alignByBasename(imdsIn, pxdsIn, strict, labelIDs)
    if nargin < 3, strict = false; end
    if nargin < 4, labelIDs = [0 255]; end

    imFiles = imdsIn.Files;
    pxFiles = pxdsIn.Files;

    imKeys = cellfun(@normKey, imFiles, 'UniformOutput', false);
    pxKeys = cellfun(@normKey, pxFiles, 'UniformOutput', false);

    [keysU, ~, idxU] = unique(pxKeys);
    firstIdx = accumarray(idxU(:), (1:numel(pxFiles))', [], @(v) v(1));
    maskMap = containers.Map(keysU, num2cell(firstIdx));

    matchedImIdx = [];
    matchedPxIdx = [];
    missingList  = {};

    for i = 1:numel(imFiles)
        k = imKeys{i};
        if isKey(maskMap, k)
            matchedImIdx(end+1) = i; %#ok<AGROW>
            matchedPxIdx(end+1) = maskMap(k); %#ok<AGROW>
        else
            missingList{end+1} = imFiles{i}; %#ok<AGROW>
        end
    end

    extraMasksIdx = setdiff(1:numel(pxFiles), matchedPxIdx);

    if strict && ~isempty(missingList)
        exampleList = strjoin(missingList(1:min(10,end)), newline);
        error('alignByBasename:MissingMasks', ...
            'Images without matching mask: %d\nExamples:\n%s', ...
            numel(missingList), exampleList);
    end

    if isempty(matchedImIdx)
        error('alignByBasename:NoPairs', 'No (image, mask) pairs found.');
    end

    imdsOut = subset(imdsIn, matchedImIdx);
    pxdsOut = pixelLabelDatastore(pxFiles(matchedPxIdx), pxdsIn.ClassNames, labelIDs);
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
    [~, name, ~] = fileparts(pathStr);
    name = lower(name);
    name = regexprep(name, '(_|-|\s)(mask|segmentation|gt|label|lbl)$', '', 'once');
    name = regexprep(name, '[^a-z0-9]+', '');
    k = name;
end


function PR = predictMaskLogical(I0, net, imageSize, useGPU)
% Returns logical mask (true=Polyp) aligned to ORIGINAL image size
    if size(I0,3)==1, I = repmat(I0,1,1,3); else, I = I0; end
    I = im2single(I);
    I = imresize(I, imageSize(1:2), "bilinear");

    X = dlarray(I, 'SSCB');
    if useGPU, X = gpuArray(X); end

    Y = predict(net, X);              % softmax HxWxCx1
    Y = gather(extractdata(Y));
    [~, idx] = max(Y, [], 3);         % 1=bg, 2=polyp
    Lsmall = idx - 1;                 % 0/1

    targetSize = size(I0); targetSize = targetSize(1:2);
    Lorig = imresize(uint8(Lsmall), targetSize, "nearest");
    PR = (Lorig == 1);                % logical
end

function [iou, dice, prec, rec, acc] = counts2metrics(TP,FP,FN,TN)
    denIoU = TP + FP + FN;
    iou  = (denIoU > 0) * (TP / max(denIoU, 1));
    dice = ((2*TP + FP + FN) > 0) * (2*TP / max(2*TP + FP + FN, 1));
    prec = TP / max(TP + FP, 1);
    rec  = TP / max(TP + FN, 1);
    acc  = (TP + TN) / max(TP + FP + FN + TN, 1);
end



function pb = makeProgressBar(N, enable, titleStr)
% GUI waitbar si hay desktop; si no, barra en consola.
    if nargin < 3, titleStr = 'Working...'; end
    if ~enable
        pb.update = @(i,~) [];
        pb.close  = @() [];
        return;
    end

    useGUI = usejava('desktop') && feature('ShowFigureWindows');

    if useGUI
        h = waitbar(0, sprintf('%s (0/%d)', titleStr, N), 'Name', titleStr);
        pb.update = @(i,NN) ...
            waitbar(i/NN, h, sprintf('%s (%d/%d)', titleStr, i, NN));
        pb.close  = @() safeDelete(h);
    else
        % Consola
        fprintf('%s\n', titleStr);
        pb.update = @(i,NN) fprintf('\r[%s] %3.0f%%  (%d/%d)', ...
            localBar(i,NN,30), 100*i/NN, i, NN);
        pb.close  = @() fprintf('\n');
    end

    function safeDelete(hh)
        if ishghandle(hh), try, delete(hh); catch, end, end
    end
    function s = localBar(i,NN,w)
        k = max(0, min(w, round(w*i/NN)));
        s = [repmat('#',1,k) repmat('-',1,w-k)];
    end
end
