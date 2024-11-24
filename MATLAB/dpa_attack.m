%% load data
clear all
clc
close all
load('attack_data_10k.mat');
load('constants.mat');
%% execute
mode = 1; % Use mode = 1 for running only the current script

if (mode == 1)
    datapoints2 = datapoints;
    datapoints2 = datapoints2*1000000;
    byte_to_attack = 1; % Use this to add a for loop and loop over all 16 bytes.
    % samples = size(datapoints2,1);
    samples = 10000; % 128;
elseif (mode == 0)
    datapoints2 = datapoints(num_trace_start:num_trace_stop,:);
    datapoints2 = datapoints2*1000000;
    %byte_to_attack = 6; % Use this to add a for loop and loop over all 16 bytes.
    samples = numoftraces;
elseif (mode ==2)
    datapoints2 = datapoints;
    datapoints2 = datapoints2*1000000;
    samples = size(datapoints2,1);
    
end
more off

% Prepare data
D = uint16(plaintexts_SCA(1:samples, :));
%clear aes_plaintexts byte_to_attack

% Prepare power traces
%eval(sprintf('traces = %s(1:samples, :);', datapoints2));
traces = datapoints2(1:samples,:);
%clear analyzed_traces

% Prepare keys
K = uint16(0:255);

% Calculate hypothetical intermediate values
V = zeros(samples, length(K));

for k = 1:length(K)
    for i = 1:samples
        xor_result = double(bitxor(D(i, byte_to_attack), K(k)));
        row = floor(xor_result / 16) + 1;
        col = mod(xor_result, 16) + 1;
        V(i, k) = SubBytes((row - 1) * 16 + col);
    end
end

% Calculate hypothetical power consumption
H = zeros(samples, length(K));

% for k = 1:length(K)
%     % Calculate Hamming weight for each intermediate value
%     H(:, k) = sum(dec2bin(V(:, k), 8) == '1', 2); % Convert to binary and count the number of 1s
% end
for k = 1:length(K)
    for i = 1:samples
        H(i, k) = HW(V(i, k) + 1);
    end
end

% Calculate the correlation
traceTimelength = size(traces, 2);
R = zeros(length(K), traceTimelength);

if (mode == 1)
    fprintf('Working on key guess = %d\n', key_for_matlab_computation_dec(byte_to_attack)); 
end

for key_index = 1:length(K)
    h_mean = mean(H(:, key_index));
    for j = 1:traceTimelength
        t_mean = mean(traces(:, j));
        numerator = sum((H(:, key_index) - h_mean) .* (traces(:, j) - t_mean));
        denominator = sqrt(sum((H(:, key_index) - h_mean).^2) * sum((traces(:, j) - t_mean).^2));
        R(key_index, j) = numerator / denominator;
    end
end
clear key_index k r j

[M,I] = max(abs(R(:)));
[key_row, key_col] = ind2sub(size(R),I);
key_found = key_row - 1;

s1 = 'DPA attack - experiment: ';
s3 = ' - Design under attack: Tiny AES (GitHub)';
s4 = ' - key byte = ';
s5 = num2str(byte_to_attack,'%d');
s6 = ' - Date: ';
s7 = date;
plot_title = strcat(s1,s3,s4,s5,s6,s7);
file_title = strcat('Tiny_AES_DPA_attack_','_key_byte',s5,'.png');
figure(1);

plot(0:255, max(abs(R), [], 2));

title(plot_title);
xlabel('Key Hypothesis');
ylabel('Correlation');
grid on;
print(file_title, '-dpng', '-r600');
maxcorr = R(key_row, key_col);

figure(2)
plot(1:size(R, 2), R);
xlabel('Time (samples)');
ylabel('Correlation');
title('Correlation vs Time');






