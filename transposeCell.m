function out = transposeCell(in)
    cat = {};
    for i = 1 : length(in)
        cat = [cat in{i}'];
    end
    out = cat;
end