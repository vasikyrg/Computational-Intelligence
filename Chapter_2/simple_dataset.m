clc
clear all

%Diavasma dedomenwn
data = load('airfoil_self_noise.dat');


%Normalization
for i = 1 : size(data,2)
    min_data = min(data(:,i));
    max_data = max(data(:,i));
    new_data(:,i) = (data(:,i)-min_data)/(max_data-min_data); %feature scalling
end
new_data = new_data(randperm(size(new_data,1)),:);
%assign data
Dtrn = new_data(1:floor(size(new_data,1)*0.6),:);
Dval = new_data(size(Dtrn,1)+1:size(Dtrn,1)+ceil(size(new_data,1)*0.2),:);
Dchk = new_data(size(Dtrn,1)+size(Dval,1)+1:end, :);

%Options gia genfis
opt(1) = genfisOptions('GridPartition');
opt(1).InputMembershipFunctionType = 'gbellmf';
opt(2) = genfisOptions('GridPartition');
opt(2).InputMembershipFunctionType = 'gbellmf';
opt(3) = genfisOptions('GridPartition');
opt(3).InputMembershipFunctionType = 'gbellmf';
opt(4) = genfisOptions('GridPartition');
opt(4).InputMembershipFunctionType = 'gbellmf';

opt(1).NumMembershipFunctions = 2;
opt(1).OutputMembershipFunctionType = 'constant'; %Singelton
opt(2).NumMembershipFunctions = 3;
opt(2).OutputMembershipFunctionType = 'constant';%Singelton
opt(3).NumMembershipFunctions = 2;
opt(3).OutputMembershipFunctionType = 'linear'; %Polynomial
opt(4).NumMembershipFunctions = 3;
opt(4).OutputMembershipFunctionType = 'linear';%Polynomial
fileID = fopen('Iterations_data.txt','w');
for i = 1:4
    
    %Training
    Tsk_model(i) = genfis(Dtrn(:,1:end-1),Dtrn(:,end),opt(i));
    trn_options = anfisOptions('InitialFis',Tsk_model(i),'EpochNumber',100);
    trn_options.ValidationData = [Dval(:,1:end-1) Dval(:,end)];
    [trnFis,trnError,stepSize,valFis,valError] = anfis([Dtrn(:,1:end-1) Dtrn(:,end)],trn_options);
    
    %Plots
        %MF before train
        figure;
        subplot(2,3,1)
        plotmf(Tsk_model(i),'input',1);
        xlabel('1. Frequency')
        
        subplot(2,3,2)
        plotmf(Tsk_model(i),'input',2);
        xlabel('2. Angle of attack')
        
        subplot(2,3,3)
        plotmf(Tsk_model(i),'input',3);
        xlabel('3. Chord length')
        
        subplot(2,3,4)
        plotmf(Tsk_model(i),'input',4);
        xlabel('4. Free-stream velocity')
        
        subplot(2,3,6)
        plotmf(Tsk_model(i),'input',5);
        xlabel('5. Suction side displacement thickness')
        suptitle(strcat("Tsk model_", strcat(int2str(i), " MFs before Training")));
        name =  strcat('TSK_model ',int2str(i), ' MFs before Training');
        saveas(gcf,name,'png'); 
        
        %MF after train
        
        figure;
        subplot(2,3,1)
        plotmf(valFis,'input',1);
        xlabel('1. Frequency')
        
        subplot(2,3,2)
        plotmf(valFis,'input',2);
        xlabel('2. Angle of attack')
        
        subplot(2,3,3)
        plotmf(valFis,'input',3);
        xlabel('3. Chord length')
        
        subplot(2,3,4)
        plotmf(valFis,'input',4);
        xlabel('4. Free-stream velocity')
        
        subplot(2,3,6)
        plotmf(valFis,'input',5);
        xlabel('5. Suction side displacement thickness')
        suptitle(strcat("Tsk model ", strcat(int2str(i), " MFs after Training")));
        name =  strcat('TSK_model ',int2str(i), ' MFs after Training');
        saveas(gcf,name,'png'); 
        
        %Iterations
        y_out = evalfis(valFis, Dchk(:,1:end-1)); 
        SSres = sum((Dchk(:,end) - y_out).^2);
        SStot = sum((Dchk(:,end) - mean(Dchk(:,end))).^2);
        R2 = 1- SSres/SStot;
        NMSE = 1-R2;
        RMSE = sqrt(mse(y_out,Dchk(:,end)));
        NDEI = sqrt(NMSE);
        pred_error = Dchk(:,end) - y_out;
        fileID = fopen('Iterations_data.txt','a');
        fprintf(fileID,strcat('TSK_model_',int2str(i)));
        fprintf(fileID,'\n');
        fprintf(fileID,'RMSE = %f\n NMSE = %f\n NDEI = %f\n R2 = %f\n', RMSE, NMSE, NDEI, R2);
        fprintf(fileID,'\n');
        
        
        %Learning Curve
        figure;
        plot([trnError valError], 'LineWidth',2);
        xlabel('Number of Iterations');
        ylabel('Error');
        legend('Training Error', 'Validation Error');
        title(strcat("Tsk model ", strcat(int2str(i), " Learning Curve")));
        name = strcat('TSK_model_',int2str(i), ' Learning Curve');
        saveas(gcf,name,'png'); 

        
        %Prediction Error
        figure;
        plot(pred_error, 'LineWidth',2);
        xlabel('Testing Data');
        ylabel('Error');
        title(strcat("Tsk model ", strcat(int2str(i), " Prediction Error")));
        name = strcat('TSK_model_',int2str(i), ' Prediction Error');
        saveas(gcf,name,'png'); 
   
    
end
fclose(fileID);