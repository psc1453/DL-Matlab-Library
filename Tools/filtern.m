% Self packaged filtern function
function output = filtern(inputA, inputB, shape)
    inputB = rot90(inputB, 2);
    output = convn(inputA, inputB, shape);
end

