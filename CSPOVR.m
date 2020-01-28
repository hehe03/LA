function [fTrain,fTest]= CSPOVR(Xtrain,Ytrain,Xtest,nFilters)

%% CSP for 3 classes(OVR)

%% Inputs:
% Xtrain:     nChannels * nTimeSamples * nTrials; Time domain raw EEG trials
% Xtest:     nChannels * nTimeSamples * nTrials; Time domain raw EEG trials
% Ytrain:    nTrials*1: 1,2,3, ...
% nFilters:  number of xDWAN spatial filters per fuzzy class; no filtering if nFilter = 0


%% Outputs:
% CSPtrain:   nFSs*nFilters * nTimeSamples * nTrials; Time domain training features after applying CSP filtering
% CSPtest:    nFSs*nFilters * nTimeSamples * nTrials; Time domain testing features after applying CSP filtering
Class=unique(Ytrain);
nFSs=3;
if nFilters>0
    nTrials=length(Ytrain);
    V=[];
    covs=zeros(size(Xtrain,1),size(Xtrain,1),size(Xtrain,3));
    for j=1:nTrials
        covs(:,:,j)=cov(Xtrain(:,:,j)');
    end
    C=cell(1,nFSs); %
    for i=1:nFSs
        ids=find(Ytrain==Class(i));
        C{i}=zeros(size(Xtrain,1));
        for j=1:length(ids)
            C{i}=C{i}+covs(:,:,ids(j));
        end
        C{i}=C{i}/length(ids); % mean covariance matrix of each class
    end
    for i=1:nFSs
        C0=zeros(size(C{i}));
        for j=[1:i-1 i+1:nFSs]
            C0=C0+C{j};%Rest
        end
        [V0,~] = eig(C0\C{i});%OVR
        V = cat(2,V,V0(:,1: nFilters)); % nTimeSamples * (nFSs * nFilters)
    end
    
    %%%%%%%%%%%%%%%%%%%%%%
    % apply filters to training data
    fTrain=zeros(size(Xtrain,3),size(V,2));
    fTest=zeros(size(Xtest,3),size(V,2));
    for k=1:size(Xtrain,3)
        X= V'*Xtrain(:,:,k);
        fTrain(k,:)=log10(diag(X*X')/trace(X*X'));
    end
    % apply filters to test data
    for k=1:size(Xtest,3)
        X = V'*Xtest(:,:,k);
        fTest(k,:)=log10(diag(X*X')/trace(X*X'));
    end
end