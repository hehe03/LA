clc; clearvars; close all; warning off all;
rng('default');

%% Scenario I-b

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

Class=[];
for a=1:3
    for b=a+1:4
        c=1:4;
        c(c==a)=[];
        c(c==b)=[];
        Class=cat(1,Class,[a,b,c(1),a,b,c(2)]);
        Class=cat(1,Class,[a,b,c(2),a,b,c(1)]);
    end
end


%% --------------------2. divide dataset-----------------------------------
nTrials=length(y)*3/4;
nDatasets=size(Class,1);
Accs=cell(1,nDatasets);
yidx=cell(1,nDatasets);

parfor ds=1:nDatasets
    ds
    a=Class(ds,1); b=Class(ds,2); c=Class(ds,3); d=Class(ds,6);
    Label1=[a,b,c]; Label2=[a,b,d];
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
        dist = zeros(nTrials);
        for i=1:nTrials
            for j=i+1:nTrials
                dist(i,j) = distance(CovT(:,:,i),CovT(:,:,j),'riemann');
                dist(j,i)=dist(i,j);
            end
        end
        
        for k=2:2:20
            k
            [~,cidx] = kmedioids(dist,k);
            ytTrain=yt(cidx);
            yidx{ds}(k,1:k,t)= ytTrain;
            
            idsTest=1:nTrials; idsTest(cidx)=[];
            ytTest=yt(idsTest);
            
            %% ---------------------- Raw----------------------------
            ys0=ysAll; ys0(ys0==c)=d;
           
            %% CSP
            XTrain=cat(3,Xt(:,:,cidx),XsAll); yTrain=cat(1,ytTrain,ys0);
            XtTest=Xt(:,:,idsTest);
            [fTrain,fTest]=CSPOVR(XTrain,yTrain,XtTest,6);
            model = fitcecoc(fTrain,yTrain);
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
            model = fitcecoc(fTrain,yTrain);
            yPred=predict(model,fTest);
            Accs{ds}(3,k,t)=100*mean(ytTest==yPred);
            
            %% ------------source dataset: EA----------------
            
            % Directly uses the source datasets

            %% CSP
            XTrain=cat(3,XtE(:,:,cidx),XsEAll);
            XtTestE=XtE(:,:,idsTest);
            [fTrain,fTest]=CSPOVR(XTrain,yTrain,XtTestE,6);
            model = fitcecoc(fTrain,yTrain);
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
            model = fitcecoc(fTrain,yTrain);
            yPred=predict(model,fTest);
            Accs{ds}(6,k,t)=100*mean(ytTest==yPred);
            
            %% ----------------------- LA ------------------------------
            if length(unique(ytTrain))==3
                
                Rt_class1=mean_covariances(CovT(:,:,cidx(ytTrain==a)),'logeuclid');
                Rt_class2=mean_covariances(CovT(:,:,cidx(ytTrain==b)),'logeuclid');
                Rt_class3=mean_covariances(CovT(:,:,cidx(ytTrain==d)),'logeuclid');
                XsLAll=[]; ysLAll=[];
                for s=1:nfiles-1
                    ys=ysAll((s-1)*nTrials+1:s*nTrials);
                    Xs=XsAll(:,:,(s-1)*nTrials+1:s*nTrials);
                    Xs_class1=Xs(:,:,ys==a); Rs_class1=mean_covariances(covariances(Xs_class1),'logeuclid');
                    Xs_class2=Xs(:,:,ys==b); Rs_class2=mean_covariances(covariances(Xs_class2),'logeuclid');
                    Xs_class3=Xs(:,:,ys==c); Rs_class3=mean_covariances(covariances(Xs_class3),'logeuclid');
                    
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
                    
                    A_coral=Rt_class3^(1/2)*Rs_class3^(-1/2);%+10^(-3)*eye(22);
                    Xs3=nan(size(Xs_class3,1),size(Xs_class3,2),size(Xs_class3,3));
                    for i=1:size(Xs_class3,3)
                        Xs3(:,:,i)=A_coral*Xs_class3(:,:,i);
                    end
                    
                    XsLA=cat(3,Xs1,Xs2,Xs3);
                    ysLA=[a*ones(size(Xs_class1,3),1); b*ones(size(Xs_class2,3),1); d*ones(size(Xs_class3,3),1)];
                    XsLAll=cat(3,XsLAll,XsLA);
                    ysLAll=cat(1,ysLAll,ysLA);
                end
                

                
                 %% CSP
                XTrain=cat(3,Xt(:,:,cidx),XsLAll); yTrain=cat(1,ytTrain,ysLAll);
                [fTrain,fTest]=CSPOVR(XTrain,yTrain,XtTest,6);
                model = fitcecoc(fTrain,yTrain);
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
                model = fitcecoc(fTrain,yTrain);
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
save('main7_CompClassifier.mat','Accs','Class','yidx')