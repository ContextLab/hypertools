function[gdata] = average_by_group(data, group_inds)

gdata = cell(size(group_inds));
F = size(data{1}, 2);

for i = 1:length(group_inds)
    next_T = min(cellfun(@(x)(size(x, 1)), data(group_inds{i})));    
    
    next_data = zeros(next_T, F);
    for j = 1:length(group_inds{i})
        next_data = next_data + data{group_inds{i}(j)};
    end
    next_data = next_data./length(group_inds{i});
    gdata{i} = next_data;
end