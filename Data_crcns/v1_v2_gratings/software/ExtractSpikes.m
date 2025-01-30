function [spikes, stim] = ExtractSpikes(neuralData, binWidth, varargin)

% Time is in ms.
% 
% Example:
%	[spikes, stim] = ExtractSpikes(neuralData, 100, ...
%		'TrialPeriod', [1 1000] + 160);

SET_CONSTS

trialPeriod = 'Full';
while ~isempty(varargin)
	switch upper(varargin{1})

	case 'TRIALPERIOD'
		trialPeriod = varargin{2};
		varargin(1:2) = [];

	end
end

[numTrials, numPops] = size(neuralData.spikeRasters);
numTrials = numTrials/2;

numUnits = zeros(numPops, 1);
for popIdx = 1:numPops
	numUnits(popIdx) = size(neuralData.spikeRasters{1,popIdx}, 1);
end

stim = zeros(numTrials, 1);

if ischar(trialPeriod)
	
	switch upper(trialPeriod)
	case 'DRIVEN'

		trialLength = DRIVEN_TRIAL_LENGTH;

		binnedTrialLength = floor(trialLength/binWidth);

		spikes = cell(1, numPops);
		for popIdx = 1:numPops
			spikes{popIdx} ...
				= zeros(numUnits(popIdx), binnedTrialLength, numTrials);
		end


		for trialIdx = 1:2:2*numTrials
			for popIdx = 1:numPops
				spikes{popIdx}(:,:,(trialIdx+1)/2) = BinTime( full( ...
					neuralData.spikeRasters{trialIdx,popIdx} ...
					), binWidth );
			end
			stim((trialIdx+1)/2) = neuralData.stim(trialIdx);
		end

	case 'BLANK'

		trialLength = BLANK_TRIAL_LENGTH;

		binnedTrialLength = floor(trialLength/binWidth);

		spikes = cell(1, numPops);
		for popIdx = 1:numPops
			spikes{popIdx} ...
				= zeros(numUnits(popIdx), binnedTrialLength, numTrials);
		end


		for trialIdx = 1:2:2*numTrials
			for popIdx = 1:numPops
				spikes{popIdx}(:,:,(trialIdx+1)/2) = BinTime( full( ...
					neuralData.spikeRasters{trialIdx+1,popIdx} ...
					), binWidth );
			end
		end

	otherwise

		trialLength = DRIVEN_TRIAL_LENGTH + BLANK_TRIAL_LENGTH;

		binnedTrialLength = floor(trialLength/binWidth);

		spikes = cell(1, numPops);
		for popIdx = 1:numPops
			spikes{popIdx} ...
				= zeros(numUnits(popIdx), binnedTrialLength, numTrials);
		end

		for trialIdx = 1:2:2*numTrials
			for popIdx = 1:numPops
				spikes{popIdx}(:,:,(trialIdx+1)/2) = BinTime( full( [...
					neuralData.spikeRasters{trialIdx,popIdx}, ...
					neuralData.spikeRasters{trialIdx+1,popIdx}...
					] ), binWidth );
			end
			stim((trialIdx+1)/2) = neuralData.stim(trialIdx);
		end

	end
	
else
	
	trialLength = range(trialPeriod) + 1;

	binnedTrialLength = floor(trialLength/binWidth);

	spikes = cell(1, numPops);
	for popIdx = 1:numPops
		spikes{popIdx} ...
			= zeros(numUnits(popIdx), binnedTrialLength, numTrials);
	end


	for trialIdx = 1:2:2*numTrials
		for popIdx = 1:numPops
			aux = [...
					neuralData.spikeRasters{trialIdx,popIdx}, ...
					neuralData.spikeRasters{trialIdx+1,popIdx}...
					];
			spikes{popIdx}(:,:,(trialIdx+1)/2) = BinTime( full( ...
				aux( :,trialPeriod(1):trialPeriod(end) )...
				), binWidth);
		end
		stim((trialIdx+1)/2) = neuralData.stim(trialIdx);
	end
	
end

end
