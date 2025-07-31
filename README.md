# Neural Network to predict house prices in California

## Instructions
```bash
./run.sh
```

## Sources
[dataset on Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices/data)

## Performance

| Metric | R² | RMSE (k$) |
|--------|-------|------|
| Baseline linear regression | 0.6488 | 69
| Best result so far `HousingNet2` |  0.8017 | 52
| Comparable found on medium using XGBoost | 0.8338 | 38

## TO DO 

- [x] train the model
- [x] split into modules
- [x] run a linear regression to compare with NN
- [x] track training process better (loss per 15 epochs)
- [x] add a validation split for early stopping (with patience counter if avg loss increase on val)
- [x] architecture tweaking (number of neurons and layers, activation functions)
- [x] histogram for visualisation
- [ ] what if I remove outliers (see histogram)?
- [x] correlation heatmap
- [ ] some automated architecture tweaking?
- [ ] lr tweaking & understanding
- [ ] validation split 
- [ ] understand how to check for overfitting? (beyond test/val split sets)
- [x] dropout technique
- [ ] save the model to be able to run it without training
- [ ] perf: is it possible to use Mac GPU? is torch already using it?
- [ ] feature tweaking
- [ ] XGBoost?

## Notes
[Some inspiring article](https://medium.com/@tejus05/california-housing-price-prediction-an-end-to-end-machine-learning-project-example-6d1a56c6c248) that achieves a 0.8338 R2 and 38,220$ RMSE. **Warning**, we don't have the same seed and split so it's not directly comparable (see difference in linear reg R2). But it's a good approx.

## Histogram of median house values
<img width="1488" height="898" alt="image" src="https://github.com/user-attachments/assets/8e62b475-b4ca-479f-8cf6-b48f80008308" />

## Correlation matrix
<img width="1486" height="910" alt="image" src="https://github.com/user-attachments/assets/7f9acddd-9814-4329-82c9-d62f5f840b0b" />


## Going further
- covariate shift and batch normalization (at each linear layer)
- layer sizing
- weight initialization
- residual path
- activation functions: how to pick
