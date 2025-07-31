# Neural Network to predict house prices in California

## Instructions
```bash
./run.sh
```

## Sources
[dataset on Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices/data)

## Infos

| Metric | Value |
|--------|-------|
| Baseline linear regression R² | 0.6488 |
| Best result so far | R² 0.8017 (RMSE 52k$) |

## TO DO 

- [x] train the model
- [x] split into modules
- [x] run a linear regression to compare with NN
- [x] track training process better (loss per 15 epochs)
- [x] add a validation split for early stopping (with patience counter if avg loss increase on val)
- [ ] architecture tweaking (number of neurons and layers, activation functions)
- [ ] automated architecture tweaking?
- [ ] lr tweaking & understanding
- [ ] validation split 
- [ ] understand how to check for overfitting? (beyond test set)
- [ ] dropout technique
- [ ] save the model to be able to run it without training
- [ ] perf: is it possible to use Mac GPU? is torch already using it?
- [ ] feature tweaking
- [ ] XGBoost?

## Remaining questions
- Some reading tells me that for such a model (regression type, not classification) between 300-800 epochs shall be fine. But 3,000 epochs and I still did not reach my early stopping (model keeps improving). Don't know why.


## Notes
[Some inspiring article](https://medium.com/@tejus05/california-housing-price-prediction-an-end-to-end-machine-learning-project-example-6d1a56c6c248) that achieves a 0.8338 R2 and 38,220$ RMSE. **Warning**, we don't have the same seed and split so it's not directly comparable (see difference in linear reg R2). But it's a good approx.


## Going further
- covariate shift and batch normalization (at each linear layer)
- layer sizing
- weight initialization
- residual path
- activation functions: how to pick