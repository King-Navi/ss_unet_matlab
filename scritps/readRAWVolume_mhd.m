function volume = readRAWVolume(rawFilePath, headerInfo)
    fid = fopen(rawFilePath, 'rb');
    if fid == -1
        error('Could not open RAW file: %s', rawFilePath);
    end

    % Determine precision based on ElementType
    switch headerInfo.ElementType
        case 'MET_UCHAR'
            precision = 'uint8';
        case 'MET_CHAR'
            % Added case for MET_CHAR (signed 8-bit integer)
            precision = 'int8';
        case 'MET_SHORT'
            precision = 'int16';
        case 'MET_USHORT'
            precision = 'uint16';
        case 'MET_FLOAT'
            precision = 'single';
        % Add other data types as needed
        otherwise
            error('Unsupported ElementType: %s', headerInfo.ElementType);
    end

    % Determine machine format (endianness)
    if isfield(headerInfo, 'BinaryDataByteOrderMSB') && headerInfo.BinaryDataByteOrderMSB
        machineFormat = 'ieee-be'; % Big-endian
    else
        machineFormat = 'ieee-le'; % Little-endian (default)
    end

    % Read the data
    numElements = prod(headerInfo.DimSize);
    data = fread(fid, numElements, precision, machineFormat);
    fclose(fid);

    % Reshape the data into the 3D volume
    volume = reshape(data, headerInfo.DimSize);
end