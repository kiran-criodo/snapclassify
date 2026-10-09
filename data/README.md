The Oxford-IIIT Pet dataset downloads into `data/pets/` the first time you train or evaluate on real photos
(about 800 MB; `data/pets/` is git-ignored). Never commit it or copy photos into the repo or the app.

Oxford-IIIT Pet Dataset: O. M. Parkhi, A. Vedaldi, A. Zisserman and C. V. Jawahar, "Cats and Dogs", IEEE CVPR
2012. Licensed CC BY-SA 4.0: credit it wherever the model is used (the app and the README do).

37 breeds (12 cats, 25 dogs), about 200 photos each; the authors' split is "trainval" (3,680 photos) and "test"
(3,669). `--subset 10` keeps the first 10 breeds by class index (5 cats and 5 dogs) for a CPU run.
