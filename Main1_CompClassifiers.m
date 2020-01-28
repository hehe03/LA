clc; clearvars; close all; warning off all;
rng('default');

%% semi-supervised
%% Overlapping label spaces, evaluate LA using CSP and RGTS

%% --------------------1. raw data-----------------------------------
addpath(genpath('/mnt/disk2/UCI_DataSet_Collection/JHU_MVBL/covariancetoolbox-master'))
dataFolder='./Data2a/';
files=dir([dataFolder 'A*.mat']);
nfiles=length(files);
XRaw=[]; yAll=[];
for s=1:nfiles
    load([dataFolder files(s).name]);
    XRaw=cat(3,XRaw,X);
    yAll=cat(1,yAll,y);
end
nTrials=length(y)/2;
Class=[];
for a=1:4
    b1=1:4;
    b1(a)=[];
    for b=b1(1:end)
        c1=b1;
        c1(c1==b)=[];
        for c=c1(1:2)
            Class=cat(1,Class,[a,b,c]);
        end
    end
end

%% --------------------2. divide dataset-----------------------------------
nDatasets=size(Class,1);
Accs=cell(1,nDatasets);
yidx=cell(1,nDatasets);

parfor ds=1:nDatasets
    ds
    a=Class(ds,1); b=Class(ds,2); c=Class(ds,3);
    Label1=[a,b]; Label2=[a,c];
    id1=find(ismember(yAll,Label1));
    id2=find(ismember(yAll,Label2));
    XR1=XRaw(:,:,id1); y1=yAll(id1); % dataset1: source
    XR2=XRaw(:,:,id2); y2=yAll(id2); % dataset2: target
    
    %% --------------------3. EA --------------------------------------
    XEA1=[]; XEA2=[];
    for t=1:nfiles
        X1=XR1(:,:,(t-1)*nTrials+1:t*nTrials);
        RE1=(mean(covariances(X1),3))^(-1/2);
        X2=XR2(:,:,(t-1)*nTrials+1:t*nTrials);
        RE2=(mean(covariances(X2),3))^(-1/2);
        xE1=nan(size(X1,1),size(X1,2),nTrials);
        xE2=xE1;
        for j=1:nTrials
            xE1(:,:,j)=RE1*X1(:,:,j);
            xE2(:,:,j)=RE2*X2(:,:,j);
        end
        XEA1=cat(3,XEA1,xE1);
        XEA2=cat(3,XEA2,xE2);
    end
    
    %% --------------------4. test algorithms--------------------------------------
    
    for t=1:nfiles
        yt=y2((t-1)*nTrials+1:t*nTrials);
        Xt=XR2(:,:,(t-1)*nTrials+1:t*nTrials);
        XtE=XEA2(:,:,(t-1)*nTrials+1:t*nTrials);
        
        ysAll=y1([1:(t-1)*nTrials t*nTrials+1:end]);
        XsAll=XR1(:,:,[1:(t-1)*nTrials t*nTrials+1:end]);
        XsEAll=XEA1(:,:,[1:(t-1)*nTrials t*nTrials+1:end]);
        
        %% clustering for Xt, then label the cluster centers for LA
        CovT=covariances(Xt);
        dist = zeros(nTrials);
        for i=1:nTrials
            for j=i+1:nTrials
                dist(i,j) = distance(CovT(:,:,i),CovT(:,:,j),'riemann');
                dist(j,i)=dist(i,j);
            end
        end
        
        CovS=covariances(XsAll);
        fs=Tangent_space(CovS);
        fs=fs';
        CovT=covariances(Xt);
        ft=Tangent_space(CovT);
        ft=ft';
        
        CovSE=covariances(XsEAll);
        CovTE=covariances(XtE);
        fsE=Tangent_space(CovSE);
        fsE=fsE';
        ftE=Tangent_space(CovTE);
        ftE=ftE';
        
            
        for k=2:2:20 % number of clusters
            [~,cidx] = kmedioids(dist,k);
            XtTrain=Xt(:,:,cidx);
            ytTrain=yt(cidx);
            XtTrainE=XtE(:,:,cidx);
            yidx{ds}(k,1:k,t)= ytTrain;
            
            idsTest=1:nTrials; idsTest(cidx)=[];
            ytTest=yt(idsTest);
            
            %% ---------------------- Raw----------------------------
            ys0=ysAll; ys0(ys0==b)=c;
            
            %% CSP
            XTrain=cat(3,XtTrain,XsAll); yTrain=cat(1,ytTrain,ys0);
            XtTest=Xt(:,:,idsTest);
            [fTrain,fTest]=CSPfeature(XTrain,yTrain,XtTest,6);
            model = fitcdiscr(fTrain,yTrain);
            yPred=predict(model,fTest);
            Accs{ds}(1,k,t)=100*mean(ytTest==yPred);
            
            %% mdm
            CovTrain=cat(3,CovT(:,:,cidx),CovS);
            CovTest=CovT(:,:,idsTest);
            yPred = mdm(CovTest,CovTrain,yTrain);
            Accs{ds}(2,k,t)=100*mean(ytTest==yPred);
            
            %% TS
            fTrain=cat(1,ft(cidx,:),fs);
            fTest=ft(idsTest,:);
            model = fitcsvm(fTrain,yTrain);
            yPred=predict(model,fTest);
            Accs{ds}(3,k,t)=100*mean(ytTest==yPred);
            
            %% ---------------------- EA----------------------------
            %% CSP
            XTrain=cat(3,XtTrainE,XsEAll);
            XtTestE=XtE(:,:,idsTest);
            [fTrain,fTest]=CSPfeature(XTrain,yTrain,XtTestE,6);
            model = fitcdiscr(fTrain,yTrain);
            yPred=predict(model,fTest);
            Accs{ds}(4,k,t)=100*mean(ytTest==yPred);
            
            %% mdm
            CovTrain=cat(3,CovTE(:,:,cidx),CovSE);
            CovTestE=CovTE(:,:,idsTest);
            yPred = mdm(CovTestE,CovTrain,yTrain);
            Accs{ds}(5,k,t)=100*mean(ytTest==yPred);
            
            %% TS
            fTrain=cat(1,ftE(cidx,:),fsE);
            fTest=ftE(idsTest,:);
            model = fitcsvm(fTrain,yTrain);
            yPred=predict(model,fTest);
            Accs{ds}(6,k,t)=100*mean(ytTest==yPred);    
            
            %%  ----------------------- LA ------------------------------
            if length(unique(ytTrain))==2
                if k==2
                    Rt_class1=CovT(:,:,cidx(ytTrain==a));
                    Rt_class2=CovT(:,:,cidx(ytTrain~=a));
                else
                    Rt_class1=mean_covariances(CovT(:,:,cidx(ytTrain==a)),'logeuclid');
                    Rt_class2=mean_covariances(CovT(:,:,cidx(ytTrain~=a)),'logeuclid');
                end
                XsLAll=[]; ysLAll=[];
                for s=1:nfiles-1
                    ys=ysAll((s-1)*nTrials+1:s*nTrials);
                    Xs=XsAll(:,:,(s-1)*nTrials+1:s*nTrials);
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
                    ysLA=[a*ones(size(Xs_class1,3),1);c*ones(size(Xs_class2,3),1)];
                    XsLAll=cat(3,XsLAll,XsLA);
                    ysLAll=cat(1,ysLAll,ysLA);
                end
                %% CSP
                XTrain=cat(3,XtTrain,XsLAll); yTrain=cat(1,ytTrain,ysLAll);
                [fTrain,fTest]=CSPfeature(XTrain,yTrain,XtTest,6);
                model = fitcdiscr(fTrain,yTrain);
                yPred=predict(model,fTest);
                Accs{ds}(7,k,t)=100*mean(ytTest==yPred);
                
                %% mdm
                CovSLA=covariances(XTrain);
                yPred = mdm(CovTest,CovSLA,yTrain);
                Accs{ds}(8,k,t)=100*mean(ytTest==yPred);
                
                %% TS
                fsLA=Tangent_space(CovSLA);
                fTrain=fsLA';
                fTest=ft(idsTest,:);
                model = fitcsvm(fTrain,yTrain);
                yPred=predict(model,fTest);
                Accs{ds}(9,k,t)=100*mean(ytTest==yPred);
                

            else
                Accs{ds}(7,k,t)=Accs{ds}(4,k,t); 
                Accs{ds}(8,k,t)=Accs{ds}(5,k,t); 
                Accs{ds}(9,k,t)=Accs{ds}(6,k,t); 
            end
        end
    end
end
save('main1.mat','Accs','Class','yidx')