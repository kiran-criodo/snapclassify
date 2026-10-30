# SnapClassify evaluation

Written by `python -m snapclassify.evaluate` (Oxford-IIIT Pet test split, first 10 breeds); rerun it rather than editing this file.

Test photos: 985 · breeds: 10 · top-1 88.1% · top-3 97.4%

## "Not sure" threshold

| threshold | answered "not sure" | top-1 when sure |
|---|---|---|
| 0.3 | 2.2% | 89.0% |
| 0.4 | 7.4% | 91.2% |
| 0.5 | 14.1% | 93.3% |
| 0.6 | 22.5% | 95.4% |
| 0.7 | 34.0% | 96.2% |

## Per breed (worst first)

| breed | test photos | accuracy |
|---|---|---|
| American Pit Bull Terrier | 100 | 65.0% |
| British Shorthair | 100 | 78.0% |
| Beagle | 100 | 81.0% |
| Boxer | 99 | 82.8% |
| Bengal | 100 | 92.0% |
| Basset Hound | 100 | 94.0% |
| American Bulldog | 100 | 95.0% |
| Abyssinian | 98 | 98.0% |
| Birman | 100 | 98.0% |
| Bombay | 88 | 98.9% |

## Most confused (true -> predicted)

| true breed | predicted as | photos |
|---|---|---|
| American Pit Bull Terrier | American Bulldog | 24 |
| Beagle | Basset Hound | 14 |
| Boxer | American Bulldog | 12 |
| British Shorthair | Abyssinian | 11 |
| Bengal | Abyssinian | 8 |
