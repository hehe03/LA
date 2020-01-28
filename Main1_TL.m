clc; clearvars; close all; %warning off all;
rng('default');

%% Scenario I-a, use SVM classifier

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
parfor (ds=1:nDatasets,nDatasets)
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
        d = zeros(nTrials);
        for i=1:nTrials
            for j=i+1:nTrials
                d(i,j) = distance(CovT(:,:,i),CovT(:,:,j),'riemann');
                d(j,i)=d(i,j);
            end
        end
        
        
        for k=2:2:20
            [~,cidx] = kmedioids(d,k);
            idsTrain=cidx;
            ytTrain=yt(cidx);
            yidx{ds}(k,1:k,t)= ytTrain;
            
            idsTest=1:nTrials; idsTest(idsTrain)=[];
            ytTest=yt(idsTest);
            
            %% ---------------------- Raw----------------------------
            ys0=ysAll; ys0(ys0==b)=c; yTrain=cat(1,ytTrain,ys0);
            fTrain=cat(1,ft(cidx,:),fs);
            ftTest=ft(idsTest,:);
            model = fitcecoc(fTrain,yTrain);
            yPred1=predict(model,ftTest);
            Accs{ds}(1,k,t)=100*mean(ytTest==yPred1);
            
            % JDA
            [acc,~,~] =MyJDA(fTrain,yTrain,ftTest,ytTest,'SVM');
            Accs{ds}(2,k,t)=100*acc;
            
            % JGSA
            [XsNew, XtNew, ~, ~] = JGSA(fTrain', ftTest', yTrain, yPred1, ytTest, []);
            SVM = fitcecoc(XsNew',yTrain);
            yPred=predict(SVM,XtNew');
            Accs{ds}(3,k,t)=100*mean(ytTest==yPred);
            
            % MEDA
            [acc,~,~,~] = MEDA(fTrain,yTrain,ftTest,ytTest,[]);
            Accs{ds}(4,k,t)=100*acc;
            
            %% ------------source dataset: EA----------------
            
            % Directly uses the source datasets
            fTrain=cat(1,ftE(cidx,:),fsE);
            ftTestE=ftE(idsTest,:);
            SVM = fitcecoc(fTrain,yTrain);
            yPred1=predict(SVM,ftTestE);
            Accs{ds}(5,k,t)=100*mean(ytTest==yPred1);
            
            % JDA
            [acc,~,~] =MyJDA(fTrain,yTrain,ftTestE,ytTest,'SVM');
            Accs{ds}(6,k,t)=100*acc;
            
            % JGSA
            [XsNew, XtNew, ~, ~] = JGSA(fTrain', ftTestE', yTrain, yPred1, ytTest, []);
            SVM = fitcecoc(XsNew',yTrain);
            yPred=predict(SVM,XtNew');
            Accs{ds}(7,k,t)=100*mean(ytTest==yPred);
            
            % MEDA
            [acc,~,~,~] = MEDA(fTrain,yTrain,ftTest,ytTest,[]);
            Accs{ds}(8,k,t)=100*acc;
            
            %% ----------------------- LA ------------------------------
            if length(unique(ytTrain))==2
                
                Rt_class1=mean_covariances(CovT(:,:,cidx(ytTrain==a)),'logeuclid');
                Rt_class2=mean_covariances(CovT(:,:,cidx(ytTrain~=a)),'logeuclid');
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
                
                yTrain=cat(1,ytTrain,ysLAll);
                CovSLA=covariances(XsLAll);
                CovTrain=cat(3,CovT(:,:,cidx),CovSLA);
                fTrain=Tangent_space(CovTrain);
                fTrain=fTrain';
                
                % LA
                SVM = fitcecoc(fTrain,yTrain);
                yPred1=predict(SVM,ftTest);
                Accs{ds}(9,k,t)=100*mean(ytTest==yPred1);
                
                % JDA
                [acc,~,~] =MyJDA(fTrain,yTrain,ftTest,ytTest,'SVM');
                Accs{ds}(10,k,t)=100*acc;
                
                % JGSA
                [XsNew, XtNew, ~, ~] = JGSA(fTrain', ftTest', yTrain, yPred1, ytTest, []);
                SVM = fitcecoc(XsNew',yTrain);
                yPred=predict(SVM,XtNew');
                Accs{ds}(11,k,t)=100*mean(ytTest==yPred);
                
                % MEDA
                [acc,~,~,~] = MEDA(fTrain,yTrain,ftTest,ytTest,[]);
                Accs{ds}(12,k,t)=100*acc;
                
            else
                Accs{ds}(9,k,t)=Accs{ds}(5,k,t);
                Accs{ds}(10,k,t)=Accs{ds}(6,k,t);
                Accs{ds}(11,k,t)=Accs{ds}(7,k,t);
                Accs{ds}(12,k,t)=Accs{ds}(8,k,t);
                
            end
        end
    end
end
save('main1_TL.mat','Accs','Class','yidx')