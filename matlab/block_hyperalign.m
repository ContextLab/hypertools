function[hdata] = block_hyperalign(data, windowlength)

T = min(cellfun(@(x)(size(x, 1)), data));

nblocks = ceil(T/windowlength);
hdata = data;
for b = 1:nblocks
    next_inds = ((b-1)*windowlength + 1):min([(b*windowlength) T]);
    next_data = cellfun(@(x)(x(next_inds, :)), data, 'UniformOutput', false);
    next_data = hyperalign(next_data{:});
    for s = 1:length(next_data)
        hdata{s}(next_inds, :) = next_data{s};
    end
end
