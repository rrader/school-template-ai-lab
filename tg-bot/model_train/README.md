## Random Forest Training

Навчальний пакет будує класичний класифікатор емоцій на основі таблиці
ландмарок, яку генерує `model_prepare/extract_face_landmarks.py`.

### Швидкий старт

```bash
cd tg-bot
conda activate moodbot
python -m model_train.train_random_forest --input model_prepare/train_face_landmarks.csv --model-output artifacts/random_forest.joblib --label-output artifacts/label_encoder.joblib  --metrics-output artifacts/metrics.json
```

Параметри:

- `--input` – шлях до CSV з колонками `file_name`, `class_name`, `x_i`, `y_i`, `z_i`.
- `--model-output` – куди зберегти модель `RandomForestClassifier` (joblib).
- `--label-output` – файл із закодованими мітками `LabelEncoder`.
- `--metrics-output` – необовʼязковий шлях для метрик (`accuracy`, `classification_report`).
- `--test-size`, `--n-estimators`, `--max-depth`, `--random-state` – стандартні
  гіперпараметри sklearn, мають розумні значення за замовчуванням.

Скрипт друкує базові метрики та записує їх у файл, якщо вказати
`--metrics-output`. Після тренування модель та енкодер можна завантажити у
будь-який інший сервіс (наприклад, Telegram-бот) та використовувати для
прогнозування настрою.

### Інференс для однієї фотографії

Коли модель натренована, можна визначити емоцію на довільному фото:

```bash
cd tg-bot
conda activate moodbot
python -m model_train.predict_emotion path/to/photo.jpg \
  --model artifacts/random_forest.joblib \
  --labels artifacts/label_encoder.joblib \
  --json-output artifacts/last_prediction.json
```

Скрипт: 

- видобуває 478 3D-ландмарок обличчя за допомогою MediaPipe FaceMesh;
- підставляє їх у `RandomForestClassifier`, отриманий після тренування;
- друкує емоцію та TOP-K (за замовчуванням 3) ймовірностей;
- за бажанням зберігає результат у JSON (`--json-output`).


### Розділення датасету

Скрипт для розділення датасету на тренувальний та тестовий:

```bash
cd tg-bot
conda activate moodbot
python -m model_train.split_dataset \
  --input model_prepare/face_landmarks.csv \
  --train-output model_prepare/train_face_landmarks.csv \
  --test-output model_prepare/test_face_landmarks.csv \
  --test-size 0.2 \
  --random-state 42
```

Параметри:

- `--input` – шлях до вхідного CSV файлу з датасетом.
- `--train-output` – шлях для збереження тренувального датасету (за замовчуванням: `train_<input_filename>.csv`).
- `--test-output` – шлях для збереження тестового датасету (за замовчуванням: `test_<input_filename>.csv`).
- `--test-size` – частка датасету для тестування (за замовчуванням: 0.2).
- `--target-column` – назва колонки з мітками класів (за замовчуванням: `class_name`).
- `--random-state` – seed для відтворюваності результатів (за замовчуванням: 42).

### Тестування моделі

Скрипт для тестування натренованої моделі на тестовому датасеті з виводом метрик якості:

```bash
cd tg-bot
conda activate moodbot
python -m model_train.test_model  --test-input model_prepare/test_face_landmarks.csv  --model artifacts/random_forest.joblib --labels artifacts/label_encoder.joblib  --metrics-output artifacts/test_metrics.json --confusion-matrix-output artifacts/confusion_matrix.csv
```

Параметри:

- `--test-input` – шлях до тестового CSV файлу (за замовчуванням: `model_prepare/test_face_landmarks.csv`).
- `--model` – шлях до натренованої моделі (за замовчуванням: `artifacts/random_forest.joblib`).
- `--labels` – шлях до LabelEncoder (за замовчуванням: `artifacts/label_encoder.joblib`).
- `--target-column` – назва колонки з мітками класів (за замовчуванням: `class_name`).
- `--metrics-output` – необовʼязковий JSON файл для збереження метрик.
- `--confusion-matrix-output` – необовʼязковий CSV файл для збереження confusion matrix.

Скрипт виводить:

- **Загальні метрики**: Accuracy, Precision (macro/weighted), Recall (macro/weighted), F1-Score (macro/weighted).
- **Метрики по класах**: Precision, Recall, F1-Score для кожного класу.
- **Confusion Matrix**: матрицю помилок для аналізу помилок класифікації.
- **Розподіл класів**: кількість зразків кожного класу в тестовому наборі.

Всі метрики можна зберегти у JSON файл для подальшого аналізу.
