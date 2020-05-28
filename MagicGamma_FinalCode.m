% -------------------------------------------------------------------------
% LOAD THE DATA AND PREPARE TRAINING AND TEST DATASETS
% -------------------------------------------------------------------------

% read the dataset
data = readtable('magic05.xls');
summary(data);

% split the dataset into test and training sets, using stratified sampling
% to maintain the same class balance in the training and test sets
rng(1);
p = cvpartition(data.class,'holdout', .3);
training_Index = training(p);
test_Index = test(p);
training_data = data(training_Index,:);
test_data = data(test_Index,:);

% calculate summary stats
GroupStats_AllData = grpstats(data,'class',{'mean','std',@skewness});
GroupStats_TrainingData = grpstats(training_data,'class', ...
                          {'mean','std',@skewness});
GroupStats_TestData = grpstats(test_data,'class',{'mean','std',@skewness});

% calculate correlation coefficient of predictors (was used on the original
% dataset that contained 10 predictors) 
%RR = [data.fConc,data.fConc1,data.fLength,data.fWidth,data.fSize,data.fAsym,data.fM3Long,data.fM3Trans,data.fAlpha,data.fDist]
%R = corrcoef(RR)

% create separate tables for the targets and predictors used in training
colnum_target = width(data);
colnum_pred = width(data)-1;
training_target = training_data(:,colnum_target);
training_predictors = training_data(:,1:colnum_pred);

% create separate tables for the targets and predictors used in the final
% test
test_target = test_data(:,colnum_target);
test_predictors = test_data(:,1:colnum_pred);


% -------------------------------------------------------------------------
% TRAINING THE DECISION TREE
% -------------------------------------------------------------------------

% train Decision Trees with different combinations of hyper parameters and
% cross-validate with 10 folds, then calculate the cross validation error
% to find the set of hyper parameters that yield a Decision Tree with the
% lowest cross validation error

% hyper parameters to vary for grid search
grid_treesplit = linspace(1,25,25);
grid_leafsize = [1 50 100 150 200 250 300];
grid_splitcriteria = ["gdi", "deviance"];
grid_numberofpredictors = linspace(1,9,9);

% initialise variables for the grid search loop
grid_combinations = numel(grid_treesplit) * numel(grid_leafsize) ...
                    * numel(grid_splitcriteria) ...
                    * numel(grid_numberofpredictors);
DT = cell(grid_combinations,9);
n=0;

% start the timer for recording the length of time taken to run the grid
% search
tic
disp('Building Decision Tree model')

% run through multiple iterations, modifying the hyper parameters each time
% to build Decision Tree models
for a=1:numel(grid_treesplit)
    for b=1:numel(grid_leafsize)
        for c=1:numel(grid_splitcriteria)
            for d=1:numel(grid_numberofpredictors)

                % record hyper parameter combination number and store in 
                % DT cell array for reference
                n = n + 1;
                DT{n,1} = n;
                
                % fit a Decision Tree on the training data with the hyper
                % parameter combination and store in DT cell array for
                % reference
                DT{n,2} = fitctree(training_predictors,training_target,...
                    'ClassNames',{'g','h'},...
                    'MaxNumSplits',grid_treesplit(a),...
                    'MinLeafSize',grid_leafsize(b),...
                    'SplitCriterion',grid_splitcriteria(c),...
                    'NumVariablesToSample',grid_numberofpredictors(d));   

                % record the values of the parameters used to create model
                % and store in DT cell array for reference
                DT{n,3} = grid_treesplit(a);
                DT{n,4} = grid_leafsize(b);
                DT{n,5} = grid_splitcriteria(c);
                DT{n,6} = grid_numberofpredictors(d);
                
                % cross-validate the Decision Tree with 10 folds and store 
                % in DT cell array for reference
                DT{n,7} = crossval(DT{n,2});

                % calculate classification error from cross-vaidation and
                % store in DT cell array for reference
                DT{n,8} = kfoldLoss(DT{n,7});
               
                % calculate the training error of the model
                DT{n,9} = resubLoss(DT{n,2});
                
                % display message to let user know how many combinations
                % have been searched through so far
                disp(['Grid combination ', num2str(n),' complete']); 
                                
            end 
        end
    end
end

% end the timer
toc

% plot a chart to visualise how varying the tree splits and leaf size 
% affects cross validation error (using all predictors and deviance as the
% split criteria)
A = DT(:,[3,4,5,6,8,9]);
rows = find( arrayfun(@(RIDX) A{RIDX,4} == 9 && strcmp(A{RIDX,3}, 'deviance'), 1:size(A,1)) );
A2 = A(rows,:);
A3 = cell2mat(A2(:,[1,2,5,6]));
AX = repmat(grid_treesplit', [1,numel(grid_leafsize)] )';
AY = repmat(grid_leafsize', [1,numel(grid_treesplit)] );
AZ1 = reshape(A3(:,3),numel(grid_leafsize),numel(grid_treesplit));
AZ2 = reshape(A3(:,4),numel(grid_leafsize),numel(grid_treesplit));
figure
surf(AX,AY,AZ1,'FaceColor','g')
hold on
surf(AX,AY,AZ2,'FaceColor','r')
xlabel('maximum tree splits')
ylabel('minimum leaf size')
zlabel('error')
title('Errors by Maximum Tree Split and Minimum Leaf Size')
legend('cross validation','training')
hold off

% create boxplot to visualise the distribution of cross val errors by the 
% number of features sampled 
B1 = cell2mat(A(:,[4,5]));
figure
boxplot(B1(:,2),B1(:,1))
title('Cross Validation Errors by Number of Features Sampled')
xlabel('number of features sampled')
ylabel('error')

% create histogram to visualise the distribution of cross val errors by 
% split criteria
rows2 = find( arrayfun(@(RIDX) strcmp(A{RIDX,3}, 'deviance'), 1:size(A,1)) );
rows3 = find( arrayfun(@(RIDX) strcmp(A{RIDX,3}, 'gdi'), 1:size(A,1)) );
CC_deviance = A(rows2,[3,5]);
CC_gdi = A(rows3,[3,5]);
figure
histogram(cell2mat(CC_deviance(:,2)))
hold on
histogram(cell2mat(CC_gdi(:,2)))
legend('deviance','gdi')
title('Cross Validation Errors by Split Criteria')
xlabel('cross val error')
hold off

% find the row index of the Decision Tree model with the lowest cross
% validation error
DT_crossvalErrors = cell2mat(DT(:,8));
DT_mincrossvalError = min(DT_crossvalErrors);
row_index_DT_mincrossvalError = find(cell2mat(DT(:,8)) == DT_mincrossvalError);

% assign the Decision Tree model with the lowest cross validation error to 
% 'bestDT' and list the hyperparameter values of this model, and training
% and cross val error rates
bestDT_cell = DT(row_index_DT_mincrossvalError, 2); 
bestDT = bestDT_cell{1};
bestDT_P1 = DT(row_index_DT_mincrossvalError, 3);
bestDT_P2 = DT(row_index_DT_mincrossvalError, 4);
bestDT_P3 = DT(row_index_DT_mincrossvalError, 5);
bestDT_P4 = DT(row_index_DT_mincrossvalError, 6);
bestDT_E1 = DT(row_index_DT_mincrossvalError, 8);
bestDT_E2 = DT(row_index_DT_mincrossvalError, 9);

% view the best Decision Tree
view(bestDT,'mode','graph')

% plot a chart to identify the most important features for discriminating
% between classes
imp = predictorImportance(bestDT);
figure;
bar(imp);
title('Predictor Importance Estimates');
ylabel('Estimates');
xlabel('Predictors');
h = gca;
h.XTickLabel = bestDT.PredictorNames;


% -------------------------------------------------------------------------
% TRAINING THE NAIVE BAYES CLASSIFIER
% -------------------------------------------------------------------------

% Train Naive Bayes Classifier models by varying the hyper parameters for
% Predictor Distribution and Kernel Smoothing Method

% initialise variables for building models
trials = 10;
k1 = zeros(1,trials);
k2 = zeros(1,trials);
k3 = zeros(1,trials);
k4 = zeros(1,trials);
k5 = zeros(1,trials);
k6 = zeros(1,trials);
t1 = zeros(1,trials);
t2 = zeros(1,trials);
t3 = zeros(1,trials);
t4 = zeros(1,trials);
t5 = zeros(1,trials);
t6 = zeros(1,trials);
M1 = cell(1,trials);
M1_crossval = cell(trials,10);
M2 = cell(1,trials);
M2_crossval = cell(trials,10);
M3 = cell(1,trials);
M3_crossval = cell(trials,10);
M4 = cell(1,trials);
M4_crossval = cell(trials,10);
M5 = cell(1,trials);
M5_crossval = cell(trials,10);
M6 = cell(1,trials);
M6_crossval = cell(trials,10);

%--------------------------------------------------------------------------
% Using Matlab Hyperparameter optimization 
%    rng default
%    M7 = fitcnb(training_predictors,training_target,'ClassNames',{'g','h'},...
%    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',...
%    struct('AcquisitionFunctionName','expected-improvement-plus'));    
%    M7_crossval = crossval(M7);
%    Loss_opt = kfoldLoss(M7);
%    k7 = Loss_opt;
%    Loss_opt_training = resubLoss(M7);
%    t7 = Loss_opt_training; 
%--------------------------------------------------------------------------

% start the timer for recording the length of time taken to run through
% the loop
tic
disp('Building Naive Bayes model')
prior = [0.7 0.3];

% run through multiple trials to find the cross validation errors for the
% models
for i= 1:trials

    % fit Model 1 - using the Normal Distribution for all predictors
    M1{i} = fitcnb(training_predictors,training_target,'DistributionNames','normal');
    M1_crossval{i} = crossval(M1{i});
    Loss_normal = kfoldLoss(M1_crossval{i});
    k1(i) = Loss_normal;
    Loss_normal_training = resubLoss(M1{i});
    t1(i) = Loss_normal_training;

    % fit Model 2 - using the Kernel Distribution, normal smoothing for all predictors    
    M2{i} = fitcnb(training_predictors,training_target,'DistributionNames','kernel','Kernel', 'normal');
    M2_crossval{i} = crossval(M2{i});
    Loss_kernel = kfoldLoss(M2_crossval{i});
    k2(i) = Loss_kernel;
    Loss_kernel_training = resubLoss(M2{i});
    t2(i) = Loss_kernel_training;

    % fit Model 3 - using the Kernel Distribution, box smoothing for all predictors     
    M3{i} = fitcnb(training_predictors,training_target,'DistributionNames','kernel','Kernel', 'box');
    M3_crossval{i} = crossval(M2{i});
    Loss_kernel_box = kfoldLoss(M3_crossval{i});
    k3(i) = Loss_kernel_box;
    Loss_kernel_box_training = resubLoss(M3{i});
    t3(i) = Loss_kernel_box_training;
     
    % fit Model 4 - using the Kernel Distribution, epanechnikov smoothing for all predictors   
    M4{i} = fitcnb(training_predictors,training_target,'DistributionNames','kernel','Kernel', 'epanechnikov');
    M4_crossval{i} = crossval(M4{i});
    Loss_kernel_epanechnikov = kfoldLoss(M4_crossval{i});
    k4(i) = Loss_kernel_epanechnikov;
    Loss_kernel_epanechnikov_training = resubLoss(M4{i});
    t4(i) = Loss_kernel_epanechnikov_training;

    % fit Model 5 - using the Kernel Distribution, triangle smoothing for all predictors  
    M5{i} = fitcnb(training_predictors,training_target,'DistributionNames','kernel','Kernel', 'triangle');
    M5_crossval{i} = crossval(M5{i});
    Loss_kernel_triangle = kfoldLoss(M5_crossval{i});
    k5(i) = Loss_kernel_triangle;
    Loss_kernel_triangle_training = resubLoss(M5{i});
    t5(i) = Loss_kernel_triangle_training;

    % fit Model 6 - using the Normal Distribution for all predictors
    M6{i} = fitcnb(training_predictors,training_target,'DistributionNames','kernel', 'Prior', prior);
    M6_crossval{i} = crossval(M6{i});
    Loss_normal_Priors = kfoldLoss(M6_crossval{i});
    k6(i) = Loss_normal_Priors;
    Loss_normal_training_Priors = resubLoss(M6{i});
    t6(i) = Loss_normal_training_Priors;
    
    disp(['Trial ', num2str(i), ' complete'])      
    
end

%end the timer
toc

% display the fold loss as a line graph for all models
numbersPlotNB = 1:trials;
kfoldErrorPlot_nor = k1;
kfoldErrorPlot_ker_normal = k2;
kfoldErrorPlot_ker_box = k3;
kfoldErrorPlot_ker_epanechnikov = k4;
kfoldErrorPlot_ker_triangle = k5;
kfoldErrorPlot_Priors = k6;
figure
plot(numbersPlotNB, kfoldErrorPlot_nor)
hold on
plot(numbersPlotNB, kfoldErrorPlot_ker_normal)
plot(numbersPlotNB, kfoldErrorPlot_ker_box)
plot(numbersPlotNB, kfoldErrorPlot_ker_epanechnikov)
plot(numbersPlotNB, kfoldErrorPlot_ker_triangle)
plot(numbersPlotNB, kfoldErrorPlot_Priors)
legend('Loss for Model1','Loss for Model2','Loss for Model3','Loss for Model4','Loss for Model5','Loss for Model6')
xlabel('trial number')
ylabel('cross-validation classification error')
title('Number of Trials vs Cross Validation Classification Error');
hold off

% extract the model with the lowest cross validation error
NB_mincrossval = min([kfoldErrorPlot_nor, kfoldErrorPlot_ker_normal, kfoldErrorPlot_ker_box, kfoldErrorPlot_ker_epanechnikov, kfoldErrorPlot_ker_triangle, kfoldErrorPlot_Priors]);   

for j=1:trials
    
    if k1(j) == NB_mincrossval
        bestNBModel = 1; 
        bestNB = M1{j};  
    end

    if k2(j) == NB_mincrossval
        bestNBModel = 2; 
        bestNB = M2{j};  
    end
    
    if k3(j) == NB_mincrossval
        bestNBModel = 3; 
        bestNB = M3{j};  
    end    
    
    if k4(j) == NB_mincrossval
        bestNBModel = 4;
        bestNB = M4{j};
    end    
    
    if k5(j) == NB_mincrossval
        bestNBModel = 5; 
        bestNB = M5{j};  
    end    

    if k6(j) == NB_mincrossval
        bestNBModel = 6; 
        bestNB = M6{j};  
    end      
    
end


% -------------------------------------------------------------------------
% FINAL TEST
% -------------------------------------------------------------------------

% run best Decision Tree Model on test data
[bestDT_output, bestDT_score] = predict(bestDT, test_predictors);

% run best Naive Bayes Classifier Model on test data
[bestNB_output, bestNB_score] = predict(bestNB, test_predictors);

% convert the table of test targets to arrays for creating the confusion
% matrix
test_target_array = table2array(test_target);

% create the confusion matrix of the best Decision Tree Model
DT_confusionMatrix = confusionmat(test_target_array,bestDT_output);
figure
DT_confusionMatrixChart = confusionchart(test_target_array,bestDT_output);
title('Confusion Matrix - Decision Tree');

% create the confusion matrix of the best Naive Bayes Model
NB_confusionMatrix = confusionmat(test_target_array,bestNB_output);
figure
NB_confusionMatrixChart = confusionchart(test_target_array,bestNB_output);
title('Confusion Matrix - Naive Bayes');

% plot the ROC curves

% ROC curve for class 'h' Hadron
figure
[DT_X, DT_Y, DT_T1, DT_AUC1] = perfcurve(test_target_array, bestDT_score(:,2),'h');
plot(DT_X,DT_Y,'m');
hold on
[NB_X, NB_Y, NB_T1, NB_AUC1] = perfcurve(test_target_array, bestNB_score(:,2),'h');
plot(NB_X,NB_Y,'g')
refline(1,0);
xlabel('False positive rate');
ylabel('True positive rate');
title('ROC Curve for Classification - Hadron');
legend('Decision Tree','Naive Bayes','Reference Line');
hold off

% ROC curve for class 'g' Gamma
figure 
[DT_V, DT_W, DT_T2, DT_AUC2] = perfcurve(test_target_array, bestDT_score(:,1),'g');
plot(DT_V,DT_W,'m');
hold on
[NB_V, NB_W, NB_T2, NB_AUC2] = perfcurve(test_target_array, bestNB_score(:,1),'g');
plot(NB_V,NB_W,'g')
refline(1,0);
xlabel('False positive rate');
ylabel('True positive rate');
title('ROC Curve for Classification - Gamma');
legend('Decision Tree','Naive Bayes','Reference Line');
hold off

% display AUC
disp(['AUC of Decision Tree Classifier is', string(DT_AUC1)])
disp(['AUC of Naive Bayes Classifier is', string(NB_AUC1)])
disp(['AUC of Decision Tree Classifier is', string(DT_AUC2)])
disp(['AUC of Naive Bayes Classifier is', string(NB_AUC2)])

% calculate the accuracy and display
NB_predictions = cell2mat(bestNB_output);
NB_target = cell2mat(test_target_array);
NB_accuracy = sum(NB_predictions == NB_target) / numel(NB_target);
disp(['Accuracy of Naive Bayes Classifier is', string(NB_accuracy)]);

DT_predictions = cell2mat(bestDT_output);
DT_target = cell2mat(test_target_array);
DT_accuracy = sum(DT_predictions == DT_target) / numel(DT_target);
disp(['Accuracy of Decision Tree is', string(DT_accuracy)]);




