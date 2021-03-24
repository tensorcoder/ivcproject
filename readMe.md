# ResNet18

TRAINING SET A , B 
val_split = trial.suggest_float('val_split', 0.1, 0.3)
lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
epochs = trial.suggest_int('epochs', 20, 60, 10)

Best is trial 23 with value: 0.9955357313156128.
Number of finished trials: 25
Best trial:
  Value: 0.9955357313156128
  Params:
    val_split: 0.2795648135214237
    lr: 0.00012626261969024224
    epochs: 50

TRAINING SET A, B
val_split = trial.suggest_float('val_split', 0.2, 0.3)
lr = trial.suggest_float('lr', 1e-5, 1e-3)
epochs = trial.suggest_int('epochs', 40, 60, 10)

Trial 24 finished with value: 0.9925000071525574 and parameters: {'val_split': 0.2493832311854751, 'lr': 8.716818376685242e-05, 'epochs': 50}. Best is trial 2 with value: 1.0.
Number of finished trials: 25
Best trial:
  Value: 1.0
  Params:
    val_split: 0.21036924222116513
    lr: 0.00010848708570856601
    epochs: 40

TRAINING SET A, C
[I 2021-03-16 16:29:58,127] Trial 24 finished with value: 0.9842932224273682 and parameters: {'val_split': 0.2388760627327019, 'lr': 0.0002666281287082156, 'epochs': 40}. Best is trial 12 with value: 0.9977272748947144.
Number of finished trials: 25
Best trial:
  Value: 0.9977272748947144
  Params:
    val_split: 0.27507103836085595
    lr: 1.9057811424398227e-05
    epochs: 40


TRAINING SET B, C
[I 2021-03-20 20:15:29,695] Trial 24 finished with value: 0.982807993888855 and parameters: {'val_split': 0.2206633476226039, 'lr': 0.0003053198277359373, 'epochs': 50}. Best is trial 20 with value: 1.0.
Number of finished trials: 25
Best trial:
  Value: 1.0
  Params:
    val_split: 0.22681432258421763
    lr: 2.2824397862053592e-05
    epochs: 50

# SVM with BOW
## iterations=1
## k=200

Kernel: rbf
svm_regularization: 1.76955055
degree: 5.0
validation accuracy: 0.7475


kernel: rbf
svm_regularization 4.564...
degree 4.0 
accuracy 0.7225

kernel : rbf
reg: 2.62
degree 1
accuracy 0.7175

surf SVM 
AB
Best trial:
  Value: 0.7455470737913485
  Params:
    k: 500.0  : 500 --> 10,000 , step=500
    iterations: 15.0m  : 5-->20
    surf_number: 9000.0  : 1000 - 10,000 : 1000
    kernel: sigmoid: linear', 'rbf', 'sigmoid'
    svm-regularization: 1.4279115308137222:  1, 5
  
BC
  Number of finished trials: 50
Best trial:
  Value: 0.7564102564102564
  Params:
    k: 200.0    ('k', 100, 1000, 100)
    iterations: 15.0    'iterations', 5, 20, 5
    surf_number: 5000.0  1000, 10000, 1000
    kernel: rbf
    svm-regularization: 4.8790100294612015

BC
Number of finished trials: 50
Best trial:
  Value: 0.7390180878552972
  Params:
    k: 300.0
    iterations: 14.0
    surf_number: 6500.0
    kernel: sigmoid
    svm-regularization: 4.801728275040972



