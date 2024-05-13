combinedData = [data_train; data_test];
rng(12);

% K-fold cross-validation
k = 5;
cv = cvpartition(size(combinedData, 1), 'KFold', k);

ClusterInfluence = 0.3:0.025:0.55; % Range af influence ranges som testes
cvMSE = zeros(cv.NumTestSets, length(ClusterInfluence));
cvRMSE = zeros(cv.NumTestSets, length(ClusterInfluence));
cvMAPE = zeros(cv.NumTestSets, length(ClusterInfluence));

for i = 1:cv.NumTestSets
    fprintf('Cross-validation fold %d/%d...\n', i, cv.NumTestSets);
    
    trainIdx = cv.training(i);
    validationIdx = cv.test(i);
    dataTrain = combinedData(trainIdx, :);
    dataValidation = combinedData(validationIdx, :);
    
    for j = 1:length(ClusterInfluence)
        options = genfisOptions('SubtractiveClustering');
        options.ClusterInfluenceRange = ClusterInfluence(j);
        initialFis = genfis(dataTrain(:,1:end-1), dataTrain(:,end), options);
        
        clear anfisOptions;
        anfisTrainingOptions = anfisOptions('InitialFIS', initialFis, 'EpochNumber', 50, 'ValidationData', dataValidation);
        anfisTrainingOptions.InitialStepSize = 0.1;
        anfisTrainingOptions.StepSizeDecreaseRate = 0.95;
        anfisTrainingOptions.StepSizeIncreaseRate = 1.05;
        
        [trainedFis, trainError, ~, chkFis, chkError] = anfis(dataTrain, anfisTrainingOptions);
        
        predictedValidationOutput = evalfis(chkFis, dataValidation(:,1:end-1));
        predictedValidationOutput(predictedValidationOutput < 0) = 0;
        
        mse = mean((dataValidation(:,end) - predictedValidationOutput).^2);
        rmse = sqrt(mse);
        mape = mean(abs((dataValidation(:,end) - predictedValidationOutput) ./ dataValidation(:,end))) * 100;
        
        cvMSE(i, j) = mse;
        cvRMSE(i, j) = rmse;
        cvMAPE(i, j) = mape;
    end
end
%%
meanCvRMSE = squeeze(mean(cvRMSE, 1));
meanCvMAPE = squeeze(mean(cvMAPE, 1));

[minRMSE, bestIdx] = min(meanCvRMSE(:));

bestClusterInfluenceIdx = ind2sub(size(meanCvRMSE), bestIdx);
bestClusterInfluence = ClusterInfluence(bestClusterInfluenceIdx);

fprintf('Bedste cluster influence range: %.3f med gns. RMSE på %.4f og MAPE påm %.4f%%\n', ...
        bestClusterInfluence, minRMSE, meanCvMAPE(bestClusterInfluenceIdx));