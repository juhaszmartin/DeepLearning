Originally i started with LSTM and experimented with that, but later moved on to transformers.
The Docker and requirements.txt is set up so that transformer.ipynb can run.

I used a 5070 for training (Blackwell), it took about 3 minutes for the 2000epochs.
Accuracy was about 50% and binary accuracy (Bullush vs Bearish only, so 3-3 unions) around 70%.

The Tiny LSTM that I overtrained on a single batch sometimes doesn't converge in one run, but with 1 or two reruns it should.

I also took data as-is meaning some people included the poles of the flags and some didn't.
I removed flags with over 100 datapoints, to make the transformer training smaller and faster, and also there were clear outliers with e.g. >17000 datapoints in a single flag.

The labeling was also inconsistent - some people used unix some used ISO timestamps - a few hours was spent on the data loader.