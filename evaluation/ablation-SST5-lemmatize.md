
## ablation-lemmatize-False on 20210501-164037

```json
{"dataset": "SST5", "trigram": false, "stopword": true, "lemmatize": false, "rand_lr": false, "feature_pick": "top", "feature_drop": 0.0, "feature_size": 20000, "penalty": 0.0, "lr_fixed": 0.01, "lr_rand_coef": 0.01, "iteration": 1, "batch": 500, "epoch": 10}
```

epoch  1, train_acc 34.094101124, acc _36.108597285_, max **36.108597285**
epoch  2, train_acc 49.286048689, acc _39.502262443_, max **39.502262443**
epoch  3, train_acc 57.595973783, acc _40.000000000_, max **40.000000000**
epoch  4, train_acc 63.494850187, acc _39.728506787_, max **40.000000000**
epoch  5, train_acc 67.298689139, acc _40.226244344_, max **40.226244344**
epoch  6, train_acc 70.552434457, acc _39.954751131_, max **40.226244344**
epoch  7, train_acc 73.899812734, acc _40.904977376_, max **40.904977376**
epoch  8, train_acc 76.100187266, acc _40.950226244_, max **40.950226244**
epoch  9, train_acc 78.242041199, acc _40.723981900_, max **40.950226244**
epoch 10, train_acc 79.939138577, acc _40.497737557_, max **40.950226244**

## ablation-lemmatize-True on 20210501-164200

```json
{"dataset": "SST5", "trigram": false, "stopword": true, "lemmatize": true, "rand_lr": false, "feature_pick": "top", "feature_drop": 0.0, "feature_size": 20000, "penalty": 0.0, "lr_fixed": 0.01, "lr_rand_coef": 0.01, "iteration": 1, "batch": 500, "epoch": 10}
```

epoch  1, train_acc 34.176029963, acc _37.692307692_, max **37.692307692**
epoch  2, train_acc 49.566947566, acc _37.782805430_, max **37.782805430**
epoch  3, train_acc 57.162921348, acc _40.995475113_, max **40.995475113**
epoch  4, train_acc 62.687265918, acc _39.909502262_, max **40.995475113**
epoch  5, train_acc 66.853932584, acc _40.904977376_, max **40.995475113**
epoch  6, train_acc 70.423689139, acc _40.045248869_, max **40.995475113**
epoch  7, train_acc 72.764513109, acc _41.447963801_, max **41.447963801**
epoch  8, train_acc 75.187265918, acc _40.407239819_, max **41.447963801**
epoch  9, train_acc 77.563202247, acc _40.904977376_, max **41.447963801**
epoch 10, train_acc 79.424157303, acc _40.814479638_, max **41.447963801**
