function extract_middle_mhd()
    % /home/ivan/Documents/SS/ProstateMR/Databases/Promise12/training_data
    % /home/ivan/Documents/SS/ProstateMR/Databases/Promise12/test_data
    inputDir   = "/home/ivan/Documents/SS/ProstateMR/Databases/Promise12/training_data";   % carpeta con .mhd/.raw
    outImgDir  = fullfile("data","ProstateMR", "training_data", "slices_img");
    outMaskDir = fullfile("data","ProstateMR", "training_data", "slices_mask");
    if ~exist(outImgDir, 'dir'),  mkdir(outImgDir);  end
    if ~exist(outMaskDir, 'dir'), mkdir(outMaskDir); end

    % Procesa sólo los .mhd que NO son máscaras
    d = dir(fullfile(inputDir, "*.mhd"));
    for k = 1:numel(d)
        name = d(k).name;
        if endsWith(lower(name), "_segmentation.mhd")
            continue; % saltar máscaras; las procesamos junto al volumen base
        end

        base = erase(name, ".mhd");  % e.g., case12
        mhdPath = fullfile(inputDir, name);
        try
            [V, info] = read_mhd(mhdPath);
        catch ME
            warning("Failed to read %s: %s", mhdPath, ME.message);
            continue;
        end

        if numel(info.DimSize) < 3
            warning("%s is not a 3D volume (DimSize=%s). Skipping.", name, mat2str(info.DimSize));
            continue;
        end

        mid = floor(info.DimSize(3)/2) + 1;
        imgSlice = V(:,:,mid);

        % Guardar imagen reescalada a 0-255
        imgOutName = sprintf("%s_z%04d.png", base, mid);
        imgOutPath = fullfile(outImgDir, imgOutName);
        imwrite(normalize_to_uint8(imgSlice), imgOutPath);

        % Si existe la máscara correspondiente, procesarla
        maskBase  = base + "_segmentation";
        maskMhd   = fullfile(inputDir, maskBase + ".mhd");
        if exist(maskMhd, "file")
            try
                [M, minfo] = read_mhd(maskMhd);
                if any(minfo.DimSize ~= info.DimSize)
                    warning("Mask %s DimSize %s differs from image %s DimSize %s. Using image mid index.", ...
                        maskBase, mat2str(minfo.DimSize), base, mat2str(info.DimSize));
                end
                maskSlice = M(:,:,min(mid, size(M,3)));

                % Conservar etiquetas tal cual; si es lógico/0-1, escalar a 0/255 para visualizar
                maskOutName = sprintf("%s_z%04d.png", maskBase, mid);
                maskOutPath = fullfile(outMaskDir, maskOutName);
                maskSliceToSave = prepare_mask_slice(maskSlice);
                imwrite(maskSliceToSave, maskOutPath);
            catch ME
                warning("Failed to read/process mask %s: %s", maskMhd, ME.message);
            end
        end

        fprintf("[OK] %s -> mid z=%d saved: %s\n", base, mid, imgOutPath);
    end
end


function [V, info] = read_mhd(mhdPath)
    % Parse minimal MetaImage header and load raw data accordingly.
    info = parse_mhd_header(mhdPath);

    % Resolve data file path (relative to header dir if needed)
    headerDir = fileparts(mhdPath);
    dataFile = info.ElementDataFile;
    if ~isfile(dataFile)
        df = fullfile(headerDir, dataFile);
        if isfile(df)
            dataFile = df;
        else
            error("ElementDataFile not found: %s", info.ElementDataFile);
        end
    end

    % Determine MATLAB class and endianness
    [cls, bytesPerElem] = element_type_to_class(info.ElementType);
    if isfield(info, 'HeaderSize') && ~isempty(info.HeaderSize)
        headerSkip = info.HeaderSize;
    else
        headerSkip = 0;
    end
    if isfield(info, 'ElementByteOrderMSB') && ~isempty(info.ElementByteOrderMSB)
        isBigEndian = logical(info.ElementByteOrderMSB);
    else
        isBigEndian = false; % default little endian
    end
    machfmt = ternary(isBigEndian, 'ieee-be', 'ieee-le');

    % Read raw data
    dims = info.DimSize(:)'; % [X Y Z]
    numelExpected = prod(dims);
    fid = fopen(dataFile, 'r', machfmt);
    if fid < 0, error("Cannot open raw data: %s", dataFile); end
    cleanup = onCleanup(@() fclose(fid));

    if headerSkip > 0
        fseek(fid, headerSkip, 'bof');
    end
    A = fread(fid, numelExpected, ['*' cls]);
    if numel(A) ~= numelExpected
        error("Read %d elements but expected %d for %s", numel(A), numelExpected, dataFile);
    end
    V = reshape(A, dims); % MetaImage is typically [X Y Z]

    % Optionally permute if you want [rows cols slices] = [Y X Z]
    % MATLAB images convention is rows(Y) x cols(X). We can transpose X/Y:
    V = permute(V, [2 1 3]); % -> [Y X Z]; feels natural for imshow
end

function info = parse_mhd_header(mhdPath)
    txt = fileread(mhdPath);
    lines = regexp(txt, '\r\n|\r|\n', 'split');
    info = struct();
    for i = 1:numel(lines)
        line = strtrim(lines{i});
        if isempty(line) || startsWith(line, '#'), continue; end
        parts = regexp(line, '^\s*([^=]+?)\s*=\s*(.+)$', 'tokens', 'once');
        if isempty(parts), continue; end
        key = strtrim(parts{1});
        val = strtrim(parts{2});
        key = matlabify_key(key);

        switch lower(key)
            case {'ndims'}
                info.NDims = str2double(val);
            case {'dimsize'}
                info.DimSize = sscanf(val, '%d')';
            case {'elementspacing', 'elementspacings'}
                info.ElementSpacing = sscanf(val, '%f')';
            case {'elementsizem', 'elementsize'}
                info.ElementSize = sscanf(val, '%f')';
            case {'elementtype'}
                info.ElementType = val;
            case {'elementdatafile'}
                info.ElementDataFile = strip_quotes(val);
            case {'elementbyteordermsb'}
                info.ElementByteOrderMSB = parse_boolean(val);
            case {'headersize'}
                info.HeaderSize = str2double(val);
            otherwise
                % keep other fields if needed
                info.(matlabify_key(key)) = val;
        end
    end
    req = {'DimSize','ElementType','ElementDataFile'};
    for r = 1:numel(req)
        if ~isfield(info, req{r}) || isempty(info.(req{r}))
            error("Missing required key '%s' in %s", req{r}, mhdPath);
        end
    end
end

function s = strip_quotes(s)
    if startsWith(s, '"') && endsWith(s, '"')
        s = s(2:end-1);
    end
end

function b = parse_boolean(s)
    s = lower(strtrim(s));
    b = any(strcmp(s, {'true','1','yes'}));
end

function key = matlabify_key(key)
    key = regexprep(key, '\s+', '_');
end

function [cls, bytesPerElem] = element_type_to_class(elemType)
    % Map MetaImage ElementType to MATLAB class
    switch upper(strtrim(elemType))
        case 'MET_UCHAR',    cls = 'uint8';   bytesPerElem = 1;
        case 'MET_CHAR',     cls = 'int8';    bytesPerElem = 1;
        case 'MET_USHORT',   cls = 'uint16';  bytesPerElem = 2;
        case 'MET_SHORT',    cls = 'int16';   bytesPerElem = 2;
        case 'MET_UINT',     cls = 'uint32';  bytesPerElem = 4;
        case 'MET_INT',      cls = 'int32';   bytesPerElem = 4;
        case 'MET_ULONG',    cls = 'uint64';  bytesPerElem = 8;
        case 'MET_LONG',     cls = 'int64';   bytesPerElem = 8;
        case 'MET_FLOAT',    cls = 'single';  bytesPerElem = 4;
        case 'MET_DOUBLE',   cls = 'double';  bytesPerElem = 8;
        otherwise
            error("Unsupported ElementType: %s", elemType);
    end
end

function y = ternary(cond, a, b), if cond, y = a; else, y = b; end, end

function I8 = normalize_to_uint8(I)
    % Robust min-max to 0-255; handles constant slices
    I = double(I);
    mn = min(I(:));
    mx = max(I(:));
    if mx > mn
        I8 = uint8(255 * (I - mn) / (mx - mn));
    else
        I8 = uint8(zeros(size(I)));
    end
end

function Mout = prepare_mask_slice(M)
    % Keep labels as-is; if binary {0,1} -> scale to {0,255} for viewing
    M = double(M);
    u = unique(M(:));
    if numel(u) <= 2 && all(ismember(u, [0 1]))
        Mout = uint8(M * 255);
    elseif all(mod(u,1) == 0) && min(u) >= 0 && max(u) <= 255
        Mout = uint8(M); % likely small label set
    else
        % General case: map labels to [0..N] then scale into 0..255
        [~,~,lbl] = unique(M(:));
        lbl = reshape(lbl-1, size(M)); % start at 0
        if max(lbl(:)) == 0
            Mout = uint8(lbl);
        else
            Mout = uint8(255 * (lbl / max(lbl(:))));
        end
    end
end
