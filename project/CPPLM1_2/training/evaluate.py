import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import torch
from attr import dataclass

@dataclass
class RegressionMetrics:
    uRMSE: torch.Tensor
    RMSE: torch.Tensor
    uMSE: torch.Tensor
    R2: torch.Tensor
    r: torch.Tensor
    rho: torch.Tensor
    
    
@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    f1: float
    recall: float
    auc: float
    spearman: float


class Evaluation():

    @staticmethod
    def regression_evaluation(
        observed: torch.Tensor,
        predicted: torch.Tensor,
        min_value = None,
        max_value = None,
        mean_value = None,
        stdev_value = 5.017634289122833,
        ) -> RegressionMetrics:
        observed = observed.cpu()
        predicted = predicted.cpu()
        if not min_value:
            min_value = torch.min(observed)
        if not max_value:
            max_value = torch.max(observed)
        if not mean_value:
            mean_value = torch.mean(observed)
        if not stdev_value:
            stdev_value = torch.std(observed)
        return RegressionMetrics(
            uRMSE=np.sqrt(mean_squared_error(observed, predicted)),
            RMSE=np.sqrt(mean_squared_error(observed, predicted))*stdev_value,
            uMSE=mean_squared_error(observed, predicted),
            R2=r2_score(observed, predicted),
            r=pearsonr(observed, predicted)[0],
            rho=spearmanr(observed, predicted)[0],
        )
    
    def classification_evaluation(self):
        return
    

if __name__ == "__main__":
    observed = torch.randn(5)
    predicted = torch.randn(5)
    print(observed)
    print(predicted)
    print(Evaluation.regression_evaluation(observed, predicted))