function B = Squash(A)

dimA = size(A);

B = reshape(A, [dimA(1), dimA(2)*dimA(3)]);

end
