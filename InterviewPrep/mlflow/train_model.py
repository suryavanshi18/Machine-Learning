import pandas as pd
import numpy as np
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import mlflow.pyfunc

test_size=0.3
n_estimator=50
max_depth=6

df=pd.read_csv('data.csv')

# actual=[1,0,1,0,1]
# pred=[0,0,1,1,1]


# actual=np.array(actual)
# pred=np.array(pred)
# actual_numpy=np.array(actual)
# pred_numpy=np.array(pred)
# print(np.sum((actual_numpy==1) & (pred_numpy==1)))
# print(np.sum((actual_numpy==0) & (pred_numpy==1)))
# print(np.sum((actual_numpy==1) & (pred_numpy==0)))
# print(np.sum((actual_numpy==0) & (pred_numpy==0)))
# print(tp)
#print(df['label'].value_counts())


'''
About Random Forest

Forest-> Group of trees
Random-> Based on Boostrapped Aggregation-> Hence random sampling is made
         If we use decision tree as base algorithm then we call it random forest
         Multiple training samples are first generated then models are trained on it.
         Aggregation step is performed, for classification task -> Majority vote is taken, whereas for regression task mean or mode strategy is used.
'''

def performance(actual,pred):
    '''
    Precision->Of all the instance model predicts as positive , how many are actually positive
    Recall->Of all the positive instance in the dataset, how many did the model correctly   identify as positive.
    tp->actual true and model also predicts true
    fp->actual false and model predicts true
    fn->actual true and model predicts negative
    tn->actual false and model also predicts false
    '''
    tp,fp,fn,tn=0,0,0,0
    for i in range(len(actual)):
        if(actual.iloc[i]==1 and pred[i]==1):
            tp+=1
        if(actual.iloc[i]==0 and pred[i]==1):
            fp+=1
        if(actual.iloc[i]==1 and pred[i]==0):
            fn+=1
        if(actual.iloc[i]==0 and pred[i]==0):
            tn+=1
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    accuracy=(tp+tn)/(tp+fp+fn+tn)
    print("Precision:",precision,"\nRecall:",recall,"\nAccuracy:",accuracy)
    return recall,precision,accuracy
    # print("Precision:",precision_score(actual,pred),"\nRecall:",recall_score(actual,pred),"\nAccuracy:",accuracy_score(actual,pred))
    
if __name__=="__main__":
    data=pd.read_csv("data.csv")
    my_exp=mlflow.set_experiment("sklearn_experiment")
    mlflow.set_tracking
    with mlflow.start_run(experiment_id=my_exp.experiment_id,run_name="Custom Run"):
        train,test=train_test_split(data,test_size=test_size) 
        mlflow.log_params({"n_estimators":n_estimator,"max_depth":max_depth})  
        x_train=train.drop(["label"],axis=1)
        y_train=train["label"]
        x_test=test.drop(["label"],axis=1)
        y_test=test["label"]
        rf_model=RandomForestClassifier(n_estimators=n_estimator,max_depth=max_depth,random_state=42)
        rf_model.fit(x_train,y_train)
        prediction=rf_model.predict(x_test)
        recall,precision,accuracy=performance(y_test,prediction)
        
        mlflow.log_metrics({"Recall score":recall,
                            "Accuracy":accuracy,"Precision":precision})
        mlflow.sklearn.log_model(
            rf_model,
            name="model",
            input_example=x_train.head(1),
            registered_model_name="RandomForestClassifier"
        )


# Notes
'''
MLmodel file encapsulates all crucial information about the model, its environment, and how to load and use it. This metadata ensures consistency, reproducibility, and seamless deployment of MLflow models across various environments.

Context Managers
with statement creates context managers-> ensures certain operations are automatically performed before and after a block of code.

When an object is used as context manager within a with statement then __enter__ and __exit__ method is called.

Setup Tracking server
 
We can host a tracking server where multiple users can log and track experiments.
It also streamlines collaborative efforts of multiple users.

We have the storage component-> Stores training and evaluation artifats


mlflow ui --backend-store-uri sqlite:///my.db \
            --default-artifact-root ./artifacts-store \
            --host 127.0.0.1
            --port 5005

Creates a SQLite database my.db in the current directory, and logging requests from clients will be pointed to this database.

We also need to set up communication component-> Interacts between users and tracking server
mlflow.set_tracking_uri(uri="...")
my_exp=mlflow.set_experiment("...........")


All details are available on mlflow dashboard

conda.yaml-> Specifies the conda environment dependencies 

A model registry is a dedicated system to manage, organize, version, and track ML models and their associated metadata.Serves as a centralized repository for storing trained models.

'''
    