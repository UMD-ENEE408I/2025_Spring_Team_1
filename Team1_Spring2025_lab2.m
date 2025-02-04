%% 3.1
% Loading an audio file
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

% Downsample to 8Hz
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

% Observe a section of the audio signal corresponding to the same time period 
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


%% 3.2
% 3.2.1 - Read audio files and calculate RMS values
yM1 = audioread("M1.wav");
yM2 = audioread("M2.wav");
yM3 = audioread("M3.wav");

rmsM1 = sqrt(mean(yM1.^2,1))
rmsM2 = sqrt(mean(yM2.^2,1))
rmsM3 = sqrt(mean(yM3.^2,1))

% 3.2.2 - The RMS value of the audio picked up from microphone 1 is higher,
% meaning the audio has a greater overall amplitude and is more likely to 
% be closer.

% 3.2.3
Rxy = zeros(length(yM1)+length(yM2)-1,1);
for m = 1:length(yM1)+length(yM2)-1
    sum = 0;
    for n = 1:length(yM1)
        if(n - m + 1 >= 1 && n - m + 1 <= length(yM1))
            sum = sum + yM1(n)*yM2(n-m+1);
        end
    end
    Rxy(m) = sum;
end

% 3.2.4
% (In degrees): 90 - acot(abs(cot(M1)-cot(M2)))
% where M1 is the angle between d1 and r and M2 is the angle between
% d2 and r.
%
% M1 = acos((d1^2 + 4*r^2 - d2^2)/(4*r*d1))
% M2 = acos((-1*d1^2 + 4*r^2 + d2^2)/(4*r*d2))
% based off calculations from 
% https://www.youtube.com/watch?v=q1SLjyz1-GM&t=251s&ab_channel=FineMath

%% 3.3
% Plot the audio signal "Cafe_with_noise.wav"
[y, Fs] = audioread("Cafe_with_noise.wav"); % Read the audio file and store the signal in 'y' and sampling rate in 'Fs' 
t = linspace(0, length(y)/Fs, length(y))'; % Generate a time vector corresponding to the the sampling rate of the signal
figure();
plot(t, y) % Plot the waveform in the time domain

% Analyze signal in frequency domain to separate regions of human voice to noise
win = hann(100, 'periodic'); 
[S, F, T] = stft(y, Fs, "Window", win);
smag = mag2db(abs(S)); % Convert the magnitude of STFT to decibels

figure();
pcolor(seconds(T), F, smag) 
xlabel('Time (s)') 
ylabel('Frequency (Hz)') 
shading flat 
colorbar 
caxis(max(smag(:)) + [-60 0])

% Identified noise at 1500 Hz

% Implement a bandstop filter to remove the noise
ynew = bandstop(y, [1400, 1600], Fs);

% Verify it was removed
[Snew, Fnew, Tnew] = stft(ynew, Fs, "Window", win);
smagnew = mag2db(abs(Snew)); % Convert the magnitude of STFT to decibels

figure();
pcolor(seconds(Tnew), Fnew, smagnew) 
xlabel('Time (s)') 
ylabel('Frequency (Hz)') 
shading flat 
colorbar 
caxis(max(smagnew(:)) + [-60 0])

% Write to new audio file
audiowrite('Cafe_clean.wav', ynew, Fs)