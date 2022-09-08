Singular Value Decomposition++ built in Scikit-Surprise for Recommender - Base problem category as per Ready Tensor specifications.

- sklearn
- Scikit-Surprise
- python
- pandas
- numpy
- scikit-optimize
- flask
- nginx
- uvicorn
- docker
- recommender system

This is a Recommender System that uses Singular Value Decomposition++ (SVD++) implemented through SciKitLearn-Surprise.

The recommender starts by trying to find matrices $U,S,V$ that best represents the user-item rating matrix $A$, where $A = USV^T$. SVD++ is a matrix factorisation technique, which reduces the number of features of a dataset by reducing the space dimension from N-dimension to K-dimension. Compared to standard SVD, SVD++ takes into account implicit ratings.

The data preprocessing step includes indexing and standardization. Numerical values (ratings) are also scaled to [0,1] using min-max scaling.

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as jester, anime, book-crossing, modcloth, amazon electronics, and movies.

This Recommender System is written using Python as its programming language. ScikitLearn-Surprise and ScikitLearn is used to implement the main algorithm, evaluate the model, and preprocess the data. Numpy, pandas, and feature_engine are used for the data preprocessing steps. SciKit-Optimize was used to handle the HPT. Flask + Nginx + gunicorn are used to provide web service which includes two endpoints- /ping for health check and /infer for predictions in real time.
