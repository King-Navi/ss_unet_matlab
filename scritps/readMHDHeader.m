function [headerInfo, rawFileName] = readMHDHeader(mhdFilePath)
    fid = fopen(mhdFilePath, 'rt');
    if fid == -1
        error('Could not open MHD file: %s', mhdFilePath);
    end

    headerInfo = struct();
    rawFileName = '';

    while ~feof(fid)
        line = fgetl(fid);
        if isempty(line) || startsWith(line, '#') % Skip empty lines and comments
            continue;
        end

        parts = strsplit(line, '=');
        if numel(parts) == 2
            key = strtrim(parts{1});
            value = strtrim(parts{2});

            switch key
                case 'DimSize'
                    headerInfo.DimSize = str2num(value); %#ok<ST2NM>
                case 'ElementType'
                    headerInfo.ElementType = value;
                case 'ElementSpacing'
                    headerInfo.ElementSpacing = str2num(value); %#ok<ST2NM>
                case 'BinaryDataByteOrderMSB'
                    headerInfo.BinaryDataByteOrderMSB = strcmpi(value, 'True');
                case 'ElementDataFile'
                    rawFileName = value;
                % Add other relevant fields as needed
            end
        end
    end
    fclose(fid);
end