% this functin will take a vector and some threshold decided based on data
% it quantize that vector(power consumption) according to that data
% this to make matrix of ON OFF for edge feature

function [quantized_sample] = quantize(vec,threshold);

quantized_sample = threshold*floor(vec/threshold);

%% Finding frequency of each STATE
j = 1;i = 1;
while i<=length(quantized_sample);
t1 = quantized_sample(i);
i = i+1;

if i>length(quantized_sample)%300
    break
end

cnt=1;
while (quantized_sample(i)==t1)
    cnt = cnt+1; i = i+1;
    if i>length(quantized_sample)%300
        break
    end
end
state(j) = t1;freq(j) = cnt;
j = j+1;
end

freq_of_states = [state' freq'];

%% Identifying Redundant STATE
ind = find(freq'<=2);rem = [];
for i = 1:length(ind)
    beg = sum(freq(1:ind(i)-1))+1;
    trm = sum(freq(1:ind(i)));
    rem = [rem beg:trm];
end

%% CLEAN QUANTIZED sample
quantized_sample(rem) = [];

end
