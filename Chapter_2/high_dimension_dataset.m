clc
clear all

%Read Data
data = readmatrix('train.csv');

%Kanonikopoihsh
for i = 1 : size(data,2)-1
    min_data = min(data(:,i));
    max_data = max(data(:,i));
    new_data(:,i) = (data(:,i)-min_data)/(max_data-min_data); %feature scalling
end
new_data = cat(2,new_data,data(:,end));
new_data = new_data(randperm(size(new_data,1)),:);

%Assign data
Dtrn = new_data(1:floor(size(new_data,1)*0.6),:);
Dval = new_data(size(Dtrn,1)+1:size(Dtrn,1)+ceil(size(new_data,1)*0.2),:);
Dchk = new_data(size(Dtrn,1)+size(Dval,1)+1:end, :);

%Cross-Validation k-fold

features = [10,15,20,25]; 
ra = [0.2,0.4,0.6,0.8]; 
num_k =5;
new_data2 = cat(1,Dtrn,Dval);
error = zeros(length(features),length(ra));


for i = 1 : 4 %length(features)
    for j = 1: 4 %length(ra)
        n_features = features(i);
        n_ra = ra(j);
        shuffle = @(new_data) new_data(randperm(size(new_data,1)),:);
        validation_error = zeros(num_k);
        
        
        %k-fold cross-validation
        for t = 1: num_k
            new_data = shuffle(new_data2);
            dtrn = new_data(1:floor(size(new_data,1)*0.8),:);
            dval = new_data(size(dtrn,1)+1:end,:);
            [indexes,weights] = relieff(dtrn(:,1:end-1),dtrn(:,end),10);
             
            genfis_opt = genfisOptions('SubtractiveClustering','ClusterInfluenceRange',n_ra);
            new_fis = genfis(dtrn(:,indexes(1:n_features)),dtrn(:,end),genfis_opt);
            
            %Training Fis
            trn_options = anfisOptions('InitialFis',new_fis,'EpochNumber',100);
            trn_options.ValidationData = [dval(:,indexes(1:n_features)) dval(:,end)];
            [trnFis,trnError,stepSize,valFis,valError] = anfis([dtrn(:,indexes(1:n_features)) dtrn(:,end)],trn_options);
            
            %Prediction Error
            validation_error(t) = min(valError); 
           
        end
        %Rules
        n_rules(i,j) = size(showrule(valFis),1);
        %Error
        error(i,j) = sum(validation_error(:)) / num_k;
    end
end

% Plotting Error with Number of Feature and Number of Rules relations
figure(1)
subplot(2,2,1);
plot(ra, error(1,:))
title('Number of Feature = 10')
subplot(2,2,2);
plot(ra, error(2,:))
title('Number of Feature = 15')
subplot(2,2,3);
plot(ra, error(3,:))
title('Number of Feature = 20')
subplot(2,2,4);
plot(ra, error(4,:))
title('Number of Feature = 25')
suptitle('Error - Number of Rules relation');
saveas(gcf, 'Error - Number of Rules.png');

figure(2)
subplot(2,2,1);
plot(features, error(:, 1))
title('Number of Radius = 0.2')
subplot(2,2,2);
plot(features, error(:, 2))
title('Number of Radius = 0.4')
subplot(2,2,3);
plot(features, error(:, 3))
title('Number of Radius = 0.6')
subplot(2,2,4);
plot(features, error(:, 4))
title('Number of Radius = 0.8')
suptitle('Error - Number of Features relation');
saveas(gcf, 'Error - Number of Features.png');


