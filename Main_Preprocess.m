clc; clearvars;

% % --------------------- Data 1: MI Data 1------------------------------
% Dataset 1: http://www.bbci.de/competition/iv/desc_1.html
% 7 subjects, 100 trials in each class; 59 EEG channels

dataFolder='E:/Data/BCIcompetition_MI/Raw/Data 1/';
files=dir([dataFolder 'BCICIV_ca*.mat']);
for s=1:length(files)
    s
    load([dataFolder files(s).name]);
    EEG=.1*double(cnt);
    b=fir1(20,2*[8 30]/nfo.fs);
    EEG=filter(b,1,EEG);
    y=mrk.y'; %(-1 for class one or 1 for class two)
    nTrials=length(y);
    X=nan(size(EEG,2),300,nTrials);
    for i=1:nTrials
        X(:,:,i)=EEG(mrk.pos(i)+0.5*nfo.fs:mrk.pos(i)+3.5*nfo.fs-1,:)'; % [0.5-3.5] seconds epoch, channels*Times
    end
    save(['E:/Data/BCIcompetition_MI/MI Data/20190921/Data1/A' num2str(s) '.mat'],'X','y');

end



%% --------------------- Data 2: MI Data 2a -----------------------------
%% downsample to 100 Hz after filter
dataFolder='E:/Data/BCIcompetition_MI/Raw/BCI4_2a/';
files=dir([dataFolder '*T.gdf']);
fs=100;
for s=1:length(files)
    s
    [EEG, h] = sload([dataFolder files(s).name]); % need to enable bioSig toolbox
    EEG(:,end-2:end)=[]; % last three channels are EOG
    for i=1:size(EEG,2)
        EEG(isnan(EEG(:,i)),i)=nanmean(EEG(:,i));
    end
    b=fir1(20,2*[8 30]/h.SampleRate); 
    EEG=filter(b,1,EEG); 
    EEG=resample(EEG,fs,h.SampleRate);
    pos=round((h.EVENT.POS-1)./(h.SampleRate/fs))+1;
    ids1=pos(h.EVENT.TYP==769); % left hand
    ids2=pos(h.EVENT.TYP==770); % right hand
    ids3=pos(h.EVENT.TYP==771); % foot
    ids4=pos(h.EVENT.TYP==772); % tongue
    y=[ones(length(ids1),1); 2*ones(length(ids2),1); 3*ones(length(ids3),1); 4*ones(length(ids4),1)];
    ids=[ids1; ids2; ids3; ids4];
    X=[];
    for i=length(ids):-1:1
        X(:,:,i)=EEG(ids(i)+.5*fs:ids(i)+3.5*fs-1,:)';
    end
    [~,index]=sort(ids);
    y=y(index); X=X(:,:,index);
    save(['E:/Data/BCIcompetition_MI/MI Data/20190921/Data2a/A' num2str(s) '.mat'],'X','y');
    
end

