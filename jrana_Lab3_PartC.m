%% Plot the audio signal "Cafe_with_noise.wav"
[y, Fs] = audioread("Cafe_with_noise.wav"); % Read the audio file and store the signal in 'y' and sampling rate in 'Fs' 
t = linspace(0, length(y)/Fs, length(y))'; % Generate a time vector corresponding to the the sampling rate of the signal
plot(t, y) % Plot the waveform in the time domain
%% Analyze signal in frequency domain to separate regions of human voice to noise
win = hann(100, 'periodic'); 
[S, F, T] = stft(y, Fs, "Window", win);
smag = mag2db(abs(S)); % Convert the magnitude of STFT to decibels

pcolor(seconds(T), F, smag) 
xlabel('Time (s)') 
ylabel('Frequency (Hz)') 
shading flat 
colorbar 
clim(max(smag(:)) + [-60 0])

% Identified noise at 1500 Hz
%% Implement a bandstop filter to remove the noise
ynew = bandstop(y, [1400, 1600], Fs);

% Verify it was removed
[Snew, Fnew, Tnew] = stft(ynew, Fs, "Window", win);
smagnew = mag2db(abs(Snew)); % Convert the magnitude of STFT to decibels

pcolor(seconds(Tnew), Fnew, smagnew) 
xlabel('Time (s)') 
ylabel('Frequency (Hz)') 
shading flat 
colorbar 
clim(max(smagnew(:)) + [-60 0])
%% Write to new audio file
audiowrite('Cafe_clean.wav', ynew, Fs)