Перед запуском необходимо склонировать репозиторий ToxiCR в текущей папке:
```sh
git clone https://github.com/WSU-SEAL/ToxiCR/tree/master
```

Затем запустить файл prepare-and-classic.py. Он сохранит разбиение на используемую при обучении и оценочную выборки. Затем обучит классические модели (результат моего запуска содержится в файле res.classic.txt).

Затем необходимо запустить файл transformers.py - он обучит модель roberta. 

Основные сложности при выполнении задания возникали из-за недостатка ресурсов и времени, а также были проблемы с переносом обучения на графические мощности.
