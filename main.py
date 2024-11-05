import pandas as pd
import lightgbm as lgb
import pickle
from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score


def load_and_preprocess_data(data_path):
    # Загрузим и подготовим данные
    data = pd.read_csv(data_path)
    X = data[["Population", 'GDP ($ per capita)', 'Literacy (%)']]
    y = data['Development Index']

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    # Разделим данные на обучающие и тестовые
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=random_state)
    logger.info(f'Размер обучающего датасета: {len(X_train)}')
    logger.info(f'Размер тренировочного датасета: {len(X_test)}')
    return X_train, X_test, y_train, y_test


def normalize_data(X_train, X_test):
    # Нормализуем данные
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Данные нормализованы")

    return X_train_scaled, X_test_scaled, scaler


def create_model():
    # Создадим модель
    model = lgb.LGBMClassifier(
        verbose=-1
    )

    return model


def train_model(model, X_train, y_train):
    # Обучим модель
    model.fit(X_train, y_train)
    logger.info('Модель обучена')
    return model


def save_model(model, path="models"):
    # Сохранение модели
    models_dir = Path(path)
    models_dir.mkdir(parents=True,
                     exist_ok=True)  # Создаем директорию, если ее еще нет

    filename = models_dir / "GradientTreeBoosting.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

    logger.info(f"Модель сохранена в: {filename}")


def evaluate_model(model, X_train, y_train, cv=5):
    # Проведем кросс-валидацию
    scores = cross_val_score(model, X_train, y_train, cv=cv)
    logger.info(f'Средняя точность кросс-валидации: {scores.mean():.3f}')


def predict_and_evaluate(model, X_test, y_test):
    # Прогнозируем на тестовых данных и оцениваем F1-метрику
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    logger.info(f'F1-мера на тестовых данных: {f1:.3f}')


if __name__ == "__main__":
    # Основной блок программы
    data_path = Path(__file__).resolve().parents[0].joinpath("data", "dataset.csv")

    # Загружаем и подготавливаем данные
    X, y = load_and_preprocess_data(data_path)

    # Разделяем данные на обучающие и тестовые
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Нормализуем данные
    X_train_scaled, X_test_scaled, _ = normalize_data(X_train, X_test)

    # Создаем и обучаем модель
    model = create_model()
    trained_model = train_model(model, X_train_scaled, y_train)

    # Сохраняем модель
    save_model(trained_model)

    # Оцениваем модель с помощью кросс-валидации
    evaluate_model(trained_model, X_train_scaled, y_train)

    # Прогнозируем на тестовых данных и оцениваем точность
    predict_and_evaluate(trained_model, X_test_scaled, y_test)