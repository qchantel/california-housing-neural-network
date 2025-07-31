# Neural Network to predict house prices in California

## Sources
[dataset on Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices/data)

## Infos
baseline linear regression R²: 0.6488

## TO DO 

- [x] train the model
- [x] split into modules
- [x] run a linear regression to compare with NN
- [x] track training process better (loss per 15 epochs)
- [x] add a validation split for early stopping (with patience counter if avg loss increase on val)
- [ ] feature tweaking
- [ ] architecture tweaking (number of neurons and layers, activation functions)
- [ ] lr tweaking & understanding
- [ ] validation split 
- [ ] understand how to check for overfitting? (beyond test set)
- [ ] dropout technique
- [ ] save the model to be able to run it without training
- [ ] perf: is it possible to use Mac GPU? is torch already using it?

## Notes
[Some inspiring article](https://medium.com/@tejus05/california-housing-price-prediction-an-end-to-end-machine-learning-project-example-6d1a56c6c248) that achieves a 0.8338 R2 and 38,220$ RMSE. Warning we don't have the same seed (see difference in linear reg R2)
