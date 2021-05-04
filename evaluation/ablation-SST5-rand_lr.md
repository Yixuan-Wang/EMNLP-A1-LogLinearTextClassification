
## ablation-rand_lr-False on 20210501-163613

```json
{"dataset": "SST5", "trigram": false, "stopword": true, "lemmatize": false, "rand_lr": false, "feature_pick": "top", "feature_drop": 0.0, "feature_size": 20000, "penalty": 0.0, "lr_fixed": 0.01, "lr_rand_coef": 0.01, "iteration": 1, "batch": 500, "epoch": 10}
```

epoch  1, train_acc 33.824906367, acc _36.199095023_, max **36.199095023**
epoch  2, train_acc 49.765917603, acc _38.325791855_, max **38.325791855**
epoch  3, train_acc 57.759831461, acc _38.552036199_, max **38.552036199**
epoch  4, train_acc 63.038389513, acc _39.638009050_, max **39.638009050**
epoch  5, train_acc 67.333801498, acc _39.321266968_, max **39.638009050**
epoch  6, train_acc 71.102528090, acc _40.180995475_, max **40.180995475**
epoch  7, train_acc 73.970037453, acc _40.135746606_, max **40.180995475**
epoch  8, train_acc 76.275749064, acc _40.633484163_, max **40.633484163**
epoch  9, train_acc 77.984550562, acc _40.407239819_, max **40.633484163**
epoch 10, train_acc 80.208333333, acc _40.361990950_, max **40.633484163**

## ablation-rand_lr-True on 20210501-163741

```json
{"dataset": "SST5", "trigram": false, "stopword": true, "lemmatize": false, "rand_lr": true, "feature_pick": "top", "feature_drop": 0.0, "feature_size": 20000, "penalty": 0.0, "lr_fixed": 0.01, "lr_rand_coef": 0.01, "iteration": 1, "batch": 500, "epoch": 10}
```

epoch  1, train_acc 32.713014981, acc _33.393665158_, max **33.393665158**
epoch  2, train_acc 44.545880150, acc _36.289592760_, max **36.289592760**
epoch  3, train_acc 50.070224719, acc _36.742081448_, max **36.742081448**
epoch  4, train_acc 54.517790262, acc _38.597285068_, max **38.597285068**
epoch  5, train_acc 58.579119850, acc _38.959276018_, max **38.959276018**
epoch  6, train_acc 60.990168539, acc _39.276018100_, max **39.276018100**
epoch  7, train_acc 64.723782772, acc _39.592760181_, max **39.592760181**
epoch  8, train_acc 66.327247191, acc _39.185520362_, max **39.592760181**
epoch  9, train_acc 68.199906367, acc _39.276018100_, max **39.592760181**
epoch 10, train_acc 69.885299625, acc _39.502262443_, max **39.592760181**
