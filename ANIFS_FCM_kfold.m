combinedData = [data_train; data_test];
rng(12);

% K-fold cross-validation
k = 5;
cv = cvpartition(size(combinedData, 1), 'KFold', k);

numClustersOptions = [3,4,5]; % Antal clusters som skal testes
exponentsOptions = 1.5:0.05:1.7; % Exponenter som skal testes for hvert cluster
cvMSE = zeros(cv.NumTestSets, length(numClustersOptions), length(exponentsOptions));
cvRMSE = zeros(cv.NumTestSets, length(numClustersOptions), length(exponentsOptions));
cvMAPE = zeros(cv.NumTestSets, length(numClustersOptions), length(exponentsOptions));

for i = 1:cv.NumTestSets
    fprintf('Cross-validation fold %d/%d...\n', i, cv.NumTestSets);
    
    trainIdx = cv.training(i);
    validationIdx = cv.test(i);
    dataTrain = combinedData(trainIdx, :);
    dataValidation = combinedData(validationIdx, :);
    
    for j = 1:length(numClustersOptions)
        for k = 1:length(exponentsOptions)
            options = genfisOptions('FCMClustering');
            options.NumClusters = numClustersOptions(j);
            options.Exponent = exponentsOptions(k);
            initialFis = genfis(dataTrain(:,1:end-1), dataTrain(:,end), options);
            
            clear anfisOptions;
            anfisTrainingOptions = anfisOptions('InitialFIS', initialFis, 'EpochNumber', 200, 'ValidationData', dataValidation);
            anfisTrainingOptions.InitialStepSize = 0.1;
            anfisTrainingOptions.StepSizeDecreaseRate = 0.95;
            anfisTrainingOptions.StepSizeIncreaseRate = 1.05;
            
            [trainedFis, trainError, ~, chkFis, chkError] = anfis(dataTrain, anfisTrainingOptions);
            
            predictedValidationOutput = evalfis(chkFis, dataValidation(:,1:end-1));
            predictedValidationOutput(predictedValidationOutput < 0) = 0;
            
            mse = mean((dataValidation(:,end) - predictedValidationOutput).^2);
            rmse = sqrt(mse);
            mape = mean(abs((validationTarget - predictedOutput) ./ validationTarget)) * 100;
            
            cvMSE(i, j, k) = mse;
            cvRMSE(i, j, k) = rmse;
            cvMAPE(i,j,k) = mape;
        end
    end
end
%%
middelCvRMSE = squeeze(mean(cvRMSE, 1));
[minRMSE, bestIdx] = min(middelCvRMSE(:));
[bestClustersIdx, bestExponentsIdx] = ind2sub(size(middelCvRMSE), bestIdx);

bestNumClusters = numClustersOptions(bestClustersIdx);
bestExponent = exponentsOptions(bestExponentsIdx);

fprintf('Bedste antal clusters: %d, bedste exponent: %.2f med gns. RMSE på %.4f\n', bestNumClusters, bestExponent, minRMSE);