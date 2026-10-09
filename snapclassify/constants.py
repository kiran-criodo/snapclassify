"""Numbers shared by training (PyTorch) and serving (NumPy): change them in one place. Given."""
MEAN = (0.485, 0.456, 0.406)      # ImageNet statistics: the pretrained backbone expects inputs normalised with them
STD = (0.229, 0.224, 0.225)
RESIZE = 256                      # the shorter side is resized to this ...
CROP = 224                        # ... then the centre CROP x CROP square is kept
ATTRIBUTION = ("Breed model trained on the Oxford-IIIT Pet Dataset (O. M. Parkhi, A. Vedaldi, A. Zisserman and "
               "C. V. Jawahar, 2012), licensed CC BY-SA 4.0.")
