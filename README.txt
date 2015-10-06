CS189 HW7 READNE

Kmeans clustering is implemented in kmeans.py, and the joke recommenders are implemented in joke.py.

Kmeans can be initialized and run and visualized using: with: 

km = Kmeans(data, k)
km.initialize()
km.cluster
show_image(km, i) #where i is the ith cluster

The average recommender for the warm up may be run with: 
avgrec = AvgRecommender(data)
avgrec.recommend()
avgrec.validate(validation)

The k-nearest neighbors recommender can be run with:
rec = Recommender(data, k)
rec.recommend(queries)
rec.validate

The PCA Recommender can be run with: 
rec = PcaRecommender(data, d)
rec.pca()
rec.validate(validation)
rec.query(queries)

The latent factor model recommender can be run with: 
rec = PcaRecommender(data, d)
rec.latent_factor(nandata, l, iterations) 

Where nandata is the original data preserving the NaNs. Validation is the validation queries, and queries are the set of queries we want to run. 





