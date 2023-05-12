
## Struktura projektu ##

### katalog `scripts`: ###
- `scraper.py` - plik skrapujący dane do zadania nr 1
- `machine_learning.py` - plik do zadań nr 2-4

### katalog `data`: ###
- `war.csv` - zestaw zawierający zescrappowane dane z Reddita potrzebne do zadania nr 1
- `result.csv` - zestaw zawierający przefiltrowane dane z pliku `war.csv`, używane do zadań 2-4

### katalog `results`: ###
- `bayes.png` - plik wynikowy do zadania 2 dla modelu naive bayes - problem z danymi
- `regression.png` - plik wynikowy do zadania 2 dla modelu logistic regression - problem z danymi
- `forest.png` - plik wynikowy do zadania 2 dla modelu random forest - problem z danymi 
- `best_model.h5` - plik z zapisanym najlepszym modelem z zadania nr 4
- `model.png` - plik wynikowy dla najlepszego modelu z zadania nr 4

## Jak korzystać ##

### Do korzystania z pliku `machine_learning.py`: ###
1. Ustawić się z poziomu terminala na katalogu głównym projektu i przy pomocy komendy `python ./scripts/machine_learning.py` uruchomić program.

Może okazać się potrzebne zainstalowanie bibliotek zewnętrznych:
- pip install pandas
- pip install numpy
- pip install matplotlib
- pip install scikit-learn
- pip install keras
- pip install tensorflow
- pip install transformers
- pip install plot-keras-history

### Do korzystania z pliku `scraper.py`: ###
1. Ustawić się z poziomu terminala na katalogu głównym projektu i przy pomocy komendy `python ./scripts/scraper.py` uruchomić program.

Uwaga! W celu uruchomienia procesu scrapowania ponownie należy odkomentować linię nr 22 i podać dane weryfikacyjne do API Reddit znajdujące się w liniach 57-63

### off ###

Przy etapie 3 dużo błędów przez które nie mogłem przejść - roc curve -     # vectorize comments using tf-idf
AttributeError: 'numpy.ndarray' object has no attribute 'lower'

Przy zadaniu 5-tym nie do końca wiedziałem co robić
