clc; clearvars; close all; warning off all;
rng('default');

%% Scenario III, use SVM classifier

addpath(genpath('/mnt/disk2/UCI_DataSet_Collection/JHU_MVBL/covariancetoolbox-master'))

%% --------------------1.  datasets -----------------------------------
XR=cell(1,2); yAll=cell(1,2); XEA=cell(1,2);
nfiles=cell(1,2); nTrials=cell(1,2);

%% MI data 1
dataFolder='./Data1/';
files=dir([dataFolder 'A*.mat']);
nfiles{1}=length(files);
XR{1}=[]; yAll{1}=[]; XEA{1}=[];
SC1=[6,11:15,26:32,43:47,51:53,56];
for s=1:nfiles{1}
    load([dataFolder files(s).name]);
    X1=X(SC1,:,:);
    XR{1}=cat(3,XR{1},X1);
    yAll{1}=cat(1,yAll{1},y);
    % EA
    nTrials{1}=length(y);
    RE=(mean(covariances(X1),3))^(-1/2);
    xE=nan(size(X1,1),size(X1,2),nTrials{1});
    for j=1:nTrials{1}
        xE(:,:,j)=RE*X1(:,:,j);
    end
    XEA{1}=cat(3,XEA{1},xE);
end

%% MI data 2a
dataFolder='./Data2a/';
files=dir([dataFolder 'A*.mat']);
nfiles{2}=length(files);
XR{2}=[]; yAll{2}=[]; XEA{2}=[];
Label=[3,4];
for s=1:nfiles{2}
    load([dataFolder files(s).name]);
    id2=find(ismember(y,Label));
    X2=X(:,:,id2);
    ytmp=y(id2);
    XR{2}=cat(3,XR{2},X2);
    yAll{2}=cat(1,yAll{2},ytmp);
    % EA
    nTrials{2}=length(ytmp);
    RE=(mean(covariances(X2),3))^(-1/2);
    xE=nan(size(X2,1),size(X2,2),nTrials{2});
    for j=1:nTrials{2}
        xE(:,:,j)=RE*X2(:,:,j);
    end
    XEA{2}=cat(3,XEA{2},xE);
end

%% --------------------2. source and target--------------------------------
ST=[1,2]; % source: data 1,  target: data 2a
% ST=[2,1]; % source: data 2a,  target: data 1

% source: ST(1)
XR1=XR{ST(1)}; y1=yAll{ST(1)}; XEA1=XEA{ST(1)};
nfiles1=nfiles{ST(1)}; nTrials1=nTrials{ST(1)};
tmp=unique(y1); a=tmp(1); b=tmp(2);

% target: ST(2)
XR2=XR{ST(2)}; y2=yAll{ST(2)}; XEA2=XEA{ST(2)};
nfiles2=nfiles{ST(2)}; nTrials2=nTrials{ST(2)};
tmp=unique(y2); c=tmp(1); d=tmp(2);

%% --------------------2. test algorithms------------------------------------
Accs=cell(1,nfiles2);
yidx=cell(1,nfiles2);
parfor t=1:nfiles2
    t
    yt=y2((t-1)*nTrials2+1:t*nTrials2); % target
    Xt=XR2(:,:,(t-1)*nTrials2+1:t*nTrials2);
    XtE=XEA2(:,:,(t-1)*nTrials2+1:t*nTrials2);
    
    ysAll=y1; XsAll=XR1; XsEAll=XEA1;
    
    %% TS features
    CovS=covariances(XsAll);
    fs=Tangent_space(CovS);
    fs=fs';
    CovT=covariances(Xt);
    ft=Tangent_space(CovT);
    ft=ft';
    
    CovSE=covariances(XsEAll);
    fsE=Tangent_space(CovSE);
    fsE=fsE';
    CovTE=covariances(XtE);
    ftE=Tangent_space(CovTE);
    ftE=ftE';
    
    %% Clustering for Xt, find a point for LA
        dist = zeros(nTrials2);
        for i=1:nTrials2
            for j=i+1:nTrials2
                dist(i,j) = distance(CovT(:,:,i),CovT(:,:,j),'riemann');
                dist(j,i)=dist(i,j);
            end
        end
    
%     Xflat=reshape(CovT,[22*22,1400]); Xflat=Xflat';  % use matlab
    
    for k=2:2:20
        
                [~,cidx] = kmedioids(dist,k);
        
%         [~, ~,~, ~, cidx, ~] = kmedoids(Xflat,k,'distance',@distfun_riemann);


        ytTrain=yt(cidx);
        yidx{t}(k,1:k)= ytTrain;
        
        idsTest=1:nTrials2; idsTest(cidx)=[];
        ytTest=yt(idsTest);
        
        %% ---------------------- Raw----------------------------
        ys0=ysAll; ys0(ys0==a)=c; ys0(ys0==b)=d;
        
        fTrain=cat(1,ft(cidx,:),fs); yTrain=cat(1,ytTrain,ys0);
        ftTest=ft(idsTest,:);
        model = fitcecoc(fTrain,yTrain);
        yPred1=predict(model,ftTest);
        Accs{t}(1,k)=100*mean(ytTest==yPred1);
        
        % JDA
        [acc,~,~] =MyJDA(fTrain,yTrain,ftTest,ytTest,'SVM');
        Accs{t}(2,k)=100*acc;  
        
        % JGSA
        [XsNew, XtNew, ~, ~] = JGSA(fTrain', ftTest', yTrain, yPred1, ytTest, []);
        SVM = fitcecoc(XsNew',yTrain);
        yPred=predict(SVM,XtNew');
        Accs{t}(3,k)=100*mean(ytTest==yPred);
        
        % MEDA
        [acc,~,~,~] = MEDA(fTrain,yTrain,ftTest,ytTest,[]);
        Accs{t}(4,k)=100*acc;
        
        %% ------------source dataset: EA----------------
        
        % Directly uses the source datasets
        fTrain=cat(1,ftE(cidx,:),fsE);
        ftTestE=ftE(idsTest,:);
        SVM = fitcecoc(fTrain,yTrain);
        yPred1=predict(SVM,ftTestE);
        Accs{t}(5,k)=100*mean(ytTest==yPred1);
        
        % JDA
        [acc,~,~] =MyJDA(fTrain,yTrain,ftTestE,ytTest,'SVM');
        Accs{t}(6,k)=100*acc;
      
        % JGSA
        [XsNew, XtNew, ~, ~] = JGSA(fTrain', ftTestE', yTrain, yPred1, ytTest, []);
        SVM = fitcecoc(XsNew',yTrain);
        yPred=predict(SVM,XtNew');
        Accs{t}(7,k)=100*mean(ytTest==yPred);
        
        % MEDA
        [acc,~,~,~] = MEDA(fTrain,yTrain,ftTestE,ytTest,[]);
        Accs{t}(8,k)=100*acc;
        
        %% ----------------------- LA ------------------------------
        if length(unique(ytTrain))==2
            
            Rt_class1=mean_covariances(CovT(:,:,cidx(ytTrain==c)),'logeuclid');
            Rt_class2=mean_covariances(CovT(:,:,cidx(ytTrain==d)),'logeuclid');
            XsLAll=[]; ysLAll=[];
            for s=1:nfiles1
                ys=ysAll((s-1)*nTrials1+1:s*nTrials1);
                Xs=XsAll(:,:,(s-1)*nTrials1+1:s*nTrials1);
                Xs_class1=Xs(:,:,ys==a); Rs_class1=mean_covariances(covariances(Xs_class1),'logeuclid');
                Xs_class2=Xs(:,:,ys==b); Rs_class2=mean_covariances(covariances(Xs_class2),'logeuclid');
                
                A_coral=Rt_class1^(1/2)*Rs_class1^(-1/2);%+10^(-3)*eye(22);
                Xs1=nan(size(Xs_class1,1),size(Xs_class1,2),size(Xs_class1,3));
                for i=1:size(Xs_class1,3)
                    Xs1(:,:,i)=A_coral*Xs_class1(:,:,i);
                end
                
                A_coral=Rt_class2^(1/2)*Rs_class2^(-1/2);%+10^(-3)*eye(22);
                Xs2=nan(size(Xs_class2,1),size(Xs_class2,2),size(Xs_class2,3));
                for i=1:size(Xs_class2,3)
                    Xs2(:,:,i)=A_coral*Xs_class2(:,:,i);
                end
                XsLA=cat(3,Xs1,Xs2);
                ysLA=[c*ones(size(Xs_class1,3),1);d*ones(size(Xs_class2,3),1)];
                XsLAll=cat(3,XsLAll,XsLA);
                ysLAll=cat(1,ysLAll,ysLA);
            end
            
            yTrain=cat(1,ytTrain,ysLAll);
            CovSLA=covariances(XsLAll);
            CovTrain=cat(3,CovT(:,:,cidx),CovSLA);
            fTrain=Tangent_space(CovTrain);
            fTrain=fTrain';
            
            % LA
            SVM = fitcecoc(fTrain,yTrain);
            yPred1=predict(SVM,ftTest);
            Accs{t}(9,k)=100*mean(ytTest==yPred1);
            
            % JDA
            [acc,~,~] =MyJDA(fTrain,yTrain,ftTest,ytTest,'SVM');
            Accs{t}(10,k)=100*acc;
            
          
            % JGSA
            [XsNew, XtNew, ~, ~] = JGSA(fTrain', ftTest', yTrain, yPred1, ytTest, []);
            SVM = fitcecoc(XsNew',yTrain);
            yPred=predict(SVM,XtNew');
            Accs{t}(11,k)=100*mean(ytTest==yPred);
            
            % MEDA
        [acc,~,~,~] = MEDA(fTrain,yTrain,ftTest,ytTest,[]);
        Accs{t}(12,k)=100*acc;

        else
            Accs{t}(9,k)=Accs{t}(5,k); 
            Accs{t}(10,k)=Accs{t}(6,k); 
            Accs{t}(11,k)=Accs{t}(7,k); 
            Accs{t}(12,k)=Accs{t}(8,k);

        end
    end
end

save('main3_TL1.mat','Accs','ST','yidx')