%% Loading an audio file
[y, Fs] = audioread("human_voice.wav"); % Read the audio file and store 
% the signal in 'y' and sampling rate in 'Fs' 
Fs
% (2) Fs = 48000 so original sampling frequency is 48000 Hz.

t = linspace(0, length(y)/Fs, length(y))'; % Generate a time vector 
% corresponding to the the sampling rate of the signal
figure();
plot(t, y) % Plot the waveform in the time domain
title("original signal")
ylabel("amplitude")
xlabel("time")

%% Downsample to 8Hz
Fsnew = Fs/6;
ynew = zeros(floor(length(y)/6)); 
length(ynew)
% (5) 14208 samples were obtained after downsampling.

for i = 1:(floor(length(y)/6));
    ynew(i) = y(6*i); % take every 6th sample of y
end

tnew = linspace(0, length(ynew)/Fsnew, length(ynew))';
figure();
plot(tnew, ynew)
title("downsampled signal")
ylabel("amplitude")
xlabel("time")

%% Observe a section of the audio signal corresponding to the same time period 
figure();
subplot(1,2,1)
startpoint = 24600; 
endpoint = 24900; % choose a specific range of values in the waveform
plot(t(startpoint:endpoint,1), y(startpoint:endpoint,1))
title("original signal")
ylabel("amplitude")
xlabel("time")

subplot(1,2,2)
plot(tnew(startpoint/6:endpoint/6,1), ynew(startpoint/6:endpoint/6,1))
title("downsampled signal")
ylabel("amplitude")
xlabel("time")

% in the original signal you can see a lot of finer detail, like some low
% frequency peaks between .516 and .517 seconds. You lose all of those
% lower frequency signals when you downsample, and you can only see the
% larger waves. The spikes in the downsampled signal are also much sharper
% whereas the original signal's spikes are more sinusoidal.

