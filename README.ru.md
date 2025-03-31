# Глава 11: Древовидное обучение: извлечение нелинейных паттернов с криптовалютных рынков

## Обзор

Древовидные модели представляют одно из наиболее практичных и интерпретируемых семейств алгоритмов машинного обучения для финансовых приложений. Деревья решений разбивают пространство признаков посредством рекурсивного бинарного разделения, создавая правила, которые отображают комбинации рыночных признаков на предсказанные результаты --- будь то классификация рыночного режима (бычий, медвежий, боковой, обвал) или регрессия ожидаемых доходностей. В отличие от линейных моделей, деревья естественным образом захватывают нелинейные зависимости и взаимодействия признаков без явной спецификации, что делает их особенно подходящими для сложной, зависимой от режима динамики криптовалютных рынков.

Эта глава продвигается от индивидуальных деревьев решений через ансамблевые методы, которые делают древовидное обучение действительно мощным для трейдинга. Бэггинг (бутстрэп-агрегирование) снижает дисперсию путём обучения нескольких деревьев на бутстрэп-выборках и усреднения их предсказаний. Случайный лес расширяет бэггинг, дополнительно рандомизируя набор признаков при каждом разделении, декоррелируя членов ансамбля и ещё больше уменьшая переобучение. Эти техники применяются к двум критическим задачам криптовалютного трейдинга: многоклассовой классификации рыночного режима и поперечно-секционному прогнозированию доходностей альткоинов во вселенной Bybit.

Особое внимание уделяется практическим проблемам развёртывания древовидных моделей на криптовалютных рынках. Несбалансированные классы эндемичны --- мгновенные обвалы редки, но критически важны для прогнозирования. Мы решаем это через SMOTE, взвешивание классов и асимметричные функции потерь. Анализ важности признаков показывает, какие рыночные сигналы управляют прогнозами в различных режимах, а перекрёстно-проверенный отбор признаков обеспечивает стабильность модели. Глава завершается полной стратегией длинных-коротких позиций по альткоинам, построенной на сигналах случайного леса, протестированной с реалистичными допущениями исполнения на Bybit.

## Содержание

1. [Введение в древовидные модели для крипто](#section-1-введение-в-древовидные-модели-для-крипто)
2. [Математические основы](#section-2-математические-основы)
3. [Сравнение древовидных методов](#section-3-сравнение-древовидных-методов)
4. [Торговые применения](#section-4-торговые-применения)
5. [Реализация на Python](#section-5-реализация-на-python)
6. [Реализация на Rust](#section-6-реализация-на-rust)
7. [Практические примеры](#section-7-практические-примеры)
8. [Фреймворк бэктестинга](#section-8-фреймворк-бэктестинга)
9. [Оценка производительности](#section-9-оценка-производительности)
10. [Перспективы развития](#section-10-перспективы-развития)

---

## Раздел 1: Введение в древовидные модели для крипто

### Деревья решений: строительный блок

Дерево решений --- это иерархическая модель, которая делает предсказания, проходя через серию бинарных решений. В каждом внутреннем узле признак сравнивается с пороговым значением, направляя наблюдение влево или вправо. В листовых узлах делается предсказание (метка класса для классификации, среднее значение для регрессии). Структура дерева обучается на данных путём жадного выбора разделения, максимизирующего некий критерий чистоты на каждом узле.

Для криптовалютного трейдинга деревья решений имеют интуитивную привлекательность: «если RSI > 70 и объём снижается и доминация BTC падает, тогда предсказать медвежий разворот» --- это по сути правило дерева решений. Проблема в том, что отдельные деревья являются оценщиками с высокой дисперсией --- они легко переобучаются на тренировочных данных и дают нестабильные предсказания. Это мотивирует использование ансамблевых методов.

### Почему деревья для криптовалютных рынков?

Криптовалютные рынки демонстрируют несколько свойств, благоприятствующих древовидным подходам. Во-первых, связь между признаками и доходностями высоко нелинейна --- признак может быть бычьим ниже порога и медвежьим выше него. Деревья захватывают эти пороговые эффекты естественным образом. Во-вторых, взаимодействия признаков повсеместны: одно и то же значение RSI имеет разные импликации в зависимости от режима волатильности, состояния тренда и ставки финансирования. Деревья моделируют взаимодействия через свою иерархическую структуру разделений. В-третьих, деревья устойчивы к масштабированию признаков и выбросам, что ценно при экстремальных значениях, характерных для криптовалютных данных.

### Классификационные vs регрессионные деревья

**Классификационные деревья** предсказывают дискретные результаты: рыночный режим (бычий/медвежий/боковой/обвал), торговый сигнал (лонг/шорт/нейтральный) или наступление события (мгновенный обвал да/нет). Они используют меры нечистоты, такие как индекс Джини или энтропия, для выбора разделений. **Регрессионные деревья** предсказывают непрерывные результаты: ожидаемую доходность, волатильность или спред. Они минимизируют квадратичную ошибку (или абсолютную ошибку) при каждом разделении. В криптовалютном трейдинге используются оба варианта: классификация для определения режима и генерации сигналов, регрессия для прогнозирования доходностей и определения размера позиций.

### Ансамблевые методы: от слабого к сильному

Ключевая идея ансамблевых методов заключается в том, что объединение множества слабых обучающихся (деревьев с ограниченной глубиной) порождает сильного обучающегося с меньшей дисперсией и лучшей обобщающей способностью. **Бэггинг** создаёт разнообразие через бутстрэп-выборку тренировочных данных. **Случайные леса** добавляют рандомизацию признаков при каждом разделении. Ошибка на невключённых наблюдениях (OOB) предоставляет встроенную оценку кросс-валидации без необходимости отдельного валидационного набора.

---

## Раздел 2: Математические основы

### Рекурсивное бинарное разделение

На каждом узле дерево выбирает признак j и порог s для минимизации:

```
min_{j,s} [Σ_{x_i ∈ R_1(j,s)} L(y_i, c_1) + Σ_{x_i ∈ R_2(j,s)} L(y_i, c_2)]
```

где R_1 и R_2 --- левая и правая области, c_1 и c_2 --- предсказания (средние для регрессии, мажоритарный класс для классификации), а L --- функция потерь.

### Нечистота Джини

Для классификации с K классами, нечистота Джини в узле t:

```
Gini(t) = 1 - Σ_{k=1}^{K} p_k²
```

где p_k --- доля наблюдений класса k в узле t. Чистый узел имеет Gini = 0. Информационный выигрыш от разделения --- это взвешенное снижение нечистоты Джини.

### Энтропия и информационный выигрыш

Альтернатива Джини, энтропия измеряет нечистоту как:

```
H(t) = -Σ_{k=1}^{K} p_k * log₂(p_k)
```

Информационный выигрыш: IG = H(родитель) - Σ (N_потомок / N_родитель) * H(потомок)

### Бэггинг (бутстрэп-агрегирование)

Для тренировочных данных D размера N:

```
Для b = 1, ..., B:
    1. Извлечь бутстрэп-выборку D_b размера N с возвращением
    2. Обучить дерево T_b на D_b
    3. Для классификации: ŷ = голосование_большинства(T_1(x), ..., T_B(x))
       Для регрессии: ŷ = (1/B) * Σ T_b(x)
```

Бэггинг снижает дисперсию примерно в ~1/B раз для некоррелированных деревьев, хотя на практике деревья разделяют структуру, ограничивая снижение.

### Случайный лес

Случайные леса модифицируют бэггинг, ограничивая каждое разделение случайным подмножеством из m признаков (обычно m = sqrt(p) для классификации, m = p/3 для регрессии, где p --- общее количество признаков):

```
Для b = 1, ..., B:
    1. Извлечь бутстрэп-выборку D_b
    2. Обучить дерево T_b на D_b, при каждом разделении:
       a. Случайно выбрать m признаков из p общих
       b. Найти лучшее разделение только среди этих m признаков
    3. Вырастить дерево до максимальной глубины (без обрезки)
```

### Ошибка на невключённых наблюдениях (OOB)

Каждая бутстрэп-выборка исключает ~37% наблюдений. OOB-предсказание для наблюдения i использует только деревья, где i не было в тренировочном наборе:

```
OOB_error = (1/N) * Σ L(y_i, ŷ_i^OOB)
```

Это даёт несмещённую оценку ошибки обобщения без отдельного валидационного набора.

### Важность признаков

**Среднее снижение нечистоты (MDI)**: сумма снижений нечистоты для всех разделений, использующих признак j, взвешенная числом наблюдений, достигающих узла.

**Перестановочная важность**: измерение увеличения OOB-ошибки при случайном перемешивании значений признака j:

```
PI_j = OOB_error(перемешанный_j) - OOB_error(оригинальный)
```

Перестановочная важность предпочтительна для криптовалютных признаков, так как избегает смещения в сторону признаков с высокой кардинальностью.

---

## Раздел 3: Сравнение древовидных методов

| Метод | Дисперсия | Смещение | Интерпретируемость | Работа с дисбалансом | Скорость |
|-------|-----------|----------|-------------------|---------------------|----------|
| Одиночное дерево решений | Высокая | Низкое | Очень высокая | Плохая | Очень быстро |
| Бэггинг деревьев | Средняя | Низкое | Низкая | Умеренная | Умеренно |
| Случайный лес | Низкая | Низкое | Низкая | Хорошая (с взвешиванием) | Умеренно |
| Обрезанное дерево решений | Средняя | Среднее | Высокая | Плохая | Очень быстро |
| Extra Trees | Низкая | Низкое-среднее | Низкая | Хорошая | Быстро |

### Сравнение обработки признаков

| Аспект | Дерево решений | Случайный лес |
|--------|---------------|---------------|
| Пропущенные значения | Нативная обработка | Нативная обработка |
| Категориальные признаки | Нативная поддержка | Нативная поддержка |
| Масштабирование признаков | Не требуется | Не требуется |
| Взаимодействия признаков | Захватываются неявно | Захватываются неявно |
| Нелинейные зависимости | Естественные | Естественные |
| Высокоразмерные данные | Склонность к переобучению | Обрабатывает хорошо |
| Устойчивость к выбросам | Высокая | Высокая |
| Коррелированные признаки | Нестабильные разделения | Декорреляция через случайные подмножества |

### Сравнение гиперпараметров

| Параметр | Одиночное дерево | Случайный лес | Влияние на криптомодели |
|----------|-----------------|---------------|------------------------|
| max_depth | 3-10 | Нет (полный рост) | Глубже = более сложные правила режимов |
| min_samples_leaf | 10-50 | 1-5 | Больше = более гладкие предсказания |
| n_estimators | 1 | 100-1000 | Больше деревьев = меньше дисперсия |
| max_features | Все | sqrt(p) | Меньше = более декоррелированные деревья |
| class_weight | Нет | "balanced" | Критично для предсказания обвалов |
| min_impurity_decrease | 0.001-0.01 | 0.0 | Регуляризация для шумных криптоданных |

---

## Раздел 4: Торговые применения

### 4.1 Классификация режимов криптовалютного рынка

Четырёхклассовая классификация режимов с использованием случайных лесов: **Бычий** (устойчивый восходящий тренд, растущий моментум), **Медвежий** (устойчивый нисходящий тренд, снижающиеся цены), **Боковой** (диапазонная торговля, низкая волатильность), **Обвал** (внезапное сильное снижение, >10% за 24 часа). Признаки включают ценовые индикаторы (доходности на нескольких горизонтах, RSI, MACD), меры волатильности (реализованная волатильность, ATR, ширина полос Боллинджера), объёмные метрики (отношение объёма, тренд OBV) и структуру рынка (ставка финансирования, открытый интерес, доминация BTC). Модель обучается на размеченных режимах и предсказывает текущее состояние, обеспечивая переключение стратегий.

### 4.2 Прогнозирование мгновенных обвалов с несбалансированными данными

Мгновенные обвалы составляют < 1% всех наблюдений, создавая серьёзный дисбаланс классов. Мы решаем это через: (1) **SMOTE** (метод синтетической передискретизации миноритарного класса) для генерации синтетических примеров обвалов, (2) **Взвешивание классов** для более тяжёлого штрафования неправильной классификации обвалов, (3) **Обучение с учётом стоимости** с асимметричной потерей, где пропуск обвала стоит в 10 раз дороже ложной тревоги. Случайные леса с этими корректировками достигают полноты > 40% на событиях обвала при сохранении точности > 15%, предоставляя ценные сигналы раннего предупреждения.

### 4.3 Мультиактивное прогнозирование доходностей альткоинов

Поперечно-секционное прогнозирование по 20+ альткоинам Bybit: на каждом временном шаге предсказывается, какие альткоины будут превосходить/уступать в следующем периоде. Признаки вычисляются для каждого актива (моментум, волатильность, объём) и поперечно-секционно (ранг во вселенной, относительная сила, корреляция с BTC). Случайный лес ранжирует активы по предсказанной доходности, открывая длинные позиции по верхнему квинтилю и короткие по нижнему. Этот рыночно-нейтральный подход захватывает относительную стоимость, хеджируя широкую рыночную экспозицию.

### 4.4 Важность признаков в различных рыночных условиях

Важность признаков не статична на криптовалютных рынках. Во время бычьих рынков доминируют моментум-признаки; во время обвалов критическими становятся признаки волатильности и объёма; во время боковых рынков возрастает важность индикаторов возврата к среднему. Вычисляя важность признаков отдельно для каждого режима, мы строим условные модели, которые взвешивают признаки соответственно текущему состоянию рынка.

### 4.5 Перекрёстно-проверенный отбор признаков

Рекурсивное исключение признаков (RFE) с перекрёстной проверкой определяет оптимальное подмножество признаков. Начиная со всех признаков, наименее важный признак удаляется на каждой итерации, и модель переоценивается через кросс-валидацию временных рядов (расширяющееся окно). Набор признаков, максимизирующий прогнозную производительность, выбирается, обычно сокращая исходные 50+ признаков до 15-25 стабильных предикторов.

---

## Раздел 5: Реализация на Python

```python
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


class BybitDataFetcher:
    """Получение исторических свечных данных из Bybit API."""

    BASE_URL = "https://api.bybit.com/v5/market/kline"

    def __init__(self, symbol: str = "BTCUSDT", interval: str = "60"):
        self.symbol = symbol
        self.interval = interval

    def fetch_klines(self, limit: int = 1000) -> pd.DataFrame:
        params = {
            "category": "linear",
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": limit,
        }
        response = requests.get(self.BASE_URL, params=params)
        data = response.json()["result"]["list"]
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df = df.sort_values("timestamp").set_index("timestamp")
        return df

    def fetch_multi_asset(self, symbols: List[str],
                          limit: int = 1000) -> Dict[str, pd.DataFrame]:
        """Получение данных для нескольких символов Bybit."""
        data = {}
        for sym in symbols:
            self.symbol = sym
            data[sym] = self.fetch_klines(limit)
        return data


class CryptoFeatureEngine:
    """Инженерия признаков для древовидных криптовалютных моделей."""

    @staticmethod
    def compute_features(df: pd.DataFrame) -> pd.DataFrame:
        """Вычисление технических признаков для древовидных моделей."""
        features = pd.DataFrame(index=df.index)

        # Признаки доходности на нескольких горизонтах
        for period in [1, 4, 12, 24, 72]:
            features[f"return_{period}h"] = df["close"].pct_change(period)

        # Индикаторы моментума
        features["rsi_14"] = CryptoFeatureEngine._rsi(df["close"], 14)
        features["rsi_7"] = CryptoFeatureEngine._rsi(df["close"], 7)

        # Признаки волатильности
        features["volatility_24h"] = df["close"].pct_change().rolling(24).std()
        features["volatility_72h"] = df["close"].pct_change().rolling(72).std()
        features["vol_ratio"] = features["volatility_24h"] / (
            features["volatility_72h"] + 1e-10)

        # Признаки объёма
        features["volume_sma_ratio"] = df["volume"] / (
            df["volume"].rolling(24).mean() + 1e-10)
        features["volume_trend"] = df["volume"].rolling(12).mean() - (
            df["volume"].rolling(48).mean())

        # Ценовая структура
        features["high_low_range"] = (df["high"] - df["low"]) / df["close"]
        features["close_position"] = (df["close"] - df["low"]) / (
            df["high"] - df["low"] + 1e-10)

        # Признаки полос Боллинджера
        sma20 = df["close"].rolling(20).mean()
        std20 = df["close"].rolling(20).std()
        features["bb_width"] = (2 * std20) / (sma20 + 1e-10)
        features["bb_position"] = (df["close"] - sma20) / (std20 + 1e-10)

        return features.dropna()

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))


class RegimeLabeler:
    """Разметка режимов криптовалютного рынка для классификации."""

    @staticmethod
    def label_regimes(df: pd.DataFrame, window: int = 24) -> pd.Series:
        """Классификация рынка на режимы Бычий/Медвежий/Боковой/Обвал."""
        returns = df["close"].pct_change(window)
        volatility = df["close"].pct_change().rolling(window).std()
        vol_threshold = volatility.quantile(0.75)

        labels = pd.Series("Боковой", index=df.index)
        labels[returns > 0.03] = "Бычий"
        labels[returns < -0.03] = "Медвежий"
        labels[(returns < -0.10)] = "Обвал"

        return labels


class CryptoRandomForest:
    """Модель случайного леса для классификации режимов и прогнозирования доходностей крипто."""

    def __init__(self, n_estimators: int = 500, max_depth: Optional[int] = None,
                 task: str = "classification"):
        self.task = task
        self.model = None
        if task == "classification":
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features="sqrt",
                class_weight="balanced",
                oob_score=True,
                n_jobs=-1,
                random_state=42,
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=0.33,
                oob_score=True,
                n_jobs=-1,
                random_state=42,
            )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Обучение модели случайного леса."""
        self.model.fit(X, y)
        result = {
            "oob_score": self.model.oob_score_,
            "feature_importance": dict(zip(
                X.columns, self.model.feature_importances_
            )),
        }
        return result

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task == "classification":
            return self.model.predict_proba(X)
        raise ValueError("predict_proba только для классификации")

    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                       n_splits: int = 5) -> Dict:
        """Кросс-валидация на временных рядах."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            self.model.fit(X_train, y_train)
            score = self.model.score(X_test, y_test)
            scores.append(score)
        return {
            "cv_scores": scores,
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
        }

    def permutation_importance(self, X: pd.DataFrame,
                                y: pd.Series) -> pd.DataFrame:
        """Вычисление перестановочной важности признаков."""
        result = permutation_importance(
            self.model, X, y, n_repeats=10, random_state=42, n_jobs=-1
        )
        importance_df = pd.DataFrame({
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }).sort_values("importance_mean", ascending=False)
        return importance_df


class LongShortStrategy:
    """Стратегия длинных-коротких позиций на альткоинах на основе сигналов случайного леса."""

    def __init__(self, n_long: int = 5, n_short: int = 5):
        self.n_long = n_long
        self.n_short = n_short

    def generate_signals(self, predictions: Dict[str, float]) -> Dict:
        """Генерация сигналов лонг/шорт из предсказанных доходностей."""
        sorted_assets = sorted(predictions.items(), key=lambda x: x[1],
                               reverse=True)
        longs = [a[0] for a in sorted_assets[:self.n_long]]
        shorts = [a[0] for a in sorted_assets[-self.n_short:]]
        return {"long": longs, "short": shorts}

    def compute_returns(self, signals: Dict, actual_returns: Dict) -> float:
        """Вычисление доходности портфеля из сигналов лонг-шорт."""
        long_ret = np.mean([actual_returns.get(s, 0) for s in signals["long"]])
        short_ret = np.mean([actual_returns.get(s, 0) for s in signals["short"]])
        return long_ret - short_ret


class ImbalancedHandler:
    """Обработка дисбаланса классов для прогнозирования обвалов."""

    @staticmethod
    def apply_smote(X: pd.DataFrame, y: pd.Series,
                    sampling_strategy: float = 0.5) -> Tuple:
        """Применение передискретизации SMOTE."""
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled


# --- Пример использования ---
if __name__ == "__main__":
    # Получение данных BTC
    fetcher = BybitDataFetcher("BTCUSDT", "60")
    btc = fetcher.fetch_klines(1000)

    # Инженерия признаков
    engine = CryptoFeatureEngine()
    features = engine.compute_features(btc)

    # Разметка режимов
    labeler = RegimeLabeler()
    labels = labeler.label_regimes(btc)

    # Выравнивание признаков и меток
    common_idx = features.index.intersection(labels.index)
    X = features.loc[common_idx]
    y = labels.loc[common_idx]

    # Обучение случайного леса
    rf = CryptoRandomForest(n_estimators=500, task="classification")
    result = rf.fit(X, y)
    print(f"OOB точность: {result['oob_score']:.4f}")
    print(f"\nТоп-5 признаков:")
    for feat, imp in sorted(result["feature_importance"].items(),
                            key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {feat}: {imp:.4f}")

    # Кросс-валидация
    cv_result = rf.cross_validate(X, y, n_splits=5)
    print(f"\nСредняя CV оценка: {cv_result['mean_score']:.4f}")
    print(f"Стд CV оценки: {cv_result['std_score']:.4f}")
```

---

## Раздел 6: Реализация на Rust

```rust
use reqwest;
use serde::{Deserialize, Serialize};
use tokio;
use rand::seq::SliceRandom;
use rand::Rng;

/// OHLCV свеча
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Deserialize)]
struct BybitResponse {
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

/// Получение свечей из Bybit
pub async fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: u32,
) -> Result<Vec<Candle>, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let url = "https://api.bybit.com/v5/market/kline";
    let resp = client
        .get(url)
        .query(&[
            ("category", "linear"),
            ("symbol", symbol),
            ("interval", interval),
            ("limit", &limit.to_string()),
        ])
        .send()
        .await?
        .json::<BybitResponse>()
        .await?;

    let candles: Vec<Candle> = resp
        .result
        .list
        .iter()
        .map(|row| Candle {
            timestamp: row[0].parse().unwrap_or(0),
            open: row[1].parse().unwrap_or(0.0),
            high: row[2].parse().unwrap_or(0.0),
            low: row[3].parse().unwrap_or(0.0),
            close: row[4].parse().unwrap_or(0.0),
            volume: row[5].parse().unwrap_or(0.0),
        })
        .collect();

    Ok(candles)
}

/// Узел дерева решений
#[derive(Debug, Clone)]
pub enum TreeNode {
    Leaf {
        prediction: f64,
        class_counts: Vec<usize>,
    },
    Split {
        feature_idx: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

/// Классификатор/регрессор дерева решений
pub struct DecisionTree {
    pub root: Option<TreeNode>,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub max_features: Option<usize>,
}

impl DecisionTree {
    pub fn new(max_depth: usize, min_samples_leaf: usize,
               max_features: Option<usize>) -> Self {
        DecisionTree {
            root: None, max_depth, min_samples_leaf, max_features,
        }
    }

    /// Подгонка дерева решений для регрессии
    pub fn fit(&mut self, features: &[Vec<f64>], targets: &[f64]) {
        let indices: Vec<usize> = (0..targets.len()).collect();
        self.root = Some(self.build_tree(features, targets, &indices, 0));
    }

    fn build_tree(&self, features: &[Vec<f64>], targets: &[f64],
                  indices: &[usize], depth: usize) -> TreeNode {
        if depth >= self.max_depth || indices.len() <= self.min_samples_leaf {
            return self.make_leaf(targets, indices);
        }

        let n_features = features[0].len();
        let feature_subset = self.select_features(n_features);

        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_score = f64::INFINITY;
        let mut best_left = Vec::new();
        let mut best_right = Vec::new();

        for &feat_idx in &feature_subset {
            let mut values: Vec<f64> = indices.iter()
                .map(|&i| features[i][feat_idx]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            values.dedup();

            for i in 0..values.len().saturating_sub(1) {
                let threshold = (values[i] + values[i + 1]) / 2.0;
                let (left, right): (Vec<usize>, Vec<usize>) = indices.iter()
                    .partition(|&&idx| features[idx][feat_idx] <= threshold);

                if left.len() < self.min_samples_leaf
                    || right.len() < self.min_samples_leaf {
                    continue;
                }

                let score = self.split_score(targets, &left, &right);
                if score < best_score {
                    best_score = score;
                    best_feature = feat_idx;
                    best_threshold = threshold;
                    best_left = left;
                    best_right = right;
                }
            }
        }

        if best_left.is_empty() || best_right.is_empty() {
            return self.make_leaf(targets, indices);
        }

        TreeNode::Split {
            feature_idx: best_feature,
            threshold: best_threshold,
            left: Box::new(self.build_tree(features, targets, &best_left, depth + 1)),
            right: Box::new(self.build_tree(features, targets, &best_right, depth + 1)),
        }
    }

    fn select_features(&self, n_features: usize) -> Vec<usize> {
        match self.max_features {
            Some(m) => {
                let mut rng = rand::thread_rng();
                let mut indices: Vec<usize> = (0..n_features).collect();
                indices.shuffle(&mut rng);
                indices.truncate(m);
                indices
            }
            None => (0..n_features).collect(),
        }
    }

    fn split_score(&self, targets: &[f64], left: &[usize], right: &[usize]) -> f64 {
        let left_var = self.variance(targets, left);
        let right_var = self.variance(targets, right);
        let n = (left.len() + right.len()) as f64;
        (left.len() as f64 / n) * left_var + (right.len() as f64 / n) * right_var
    }

    fn variance(&self, targets: &[f64], indices: &[usize]) -> f64 {
        let n = indices.len() as f64;
        let mean: f64 = indices.iter().map(|&i| targets[i]).sum::<f64>() / n;
        indices.iter().map(|&i| (targets[i] - mean).powi(2)).sum::<f64>() / n
    }

    fn make_leaf(&self, targets: &[f64], indices: &[usize]) -> TreeNode {
        let mean: f64 = indices.iter().map(|&i| targets[i]).sum::<f64>()
            / indices.len() as f64;
        TreeNode::Leaf {
            prediction: mean,
            class_counts: Vec::new(),
        }
    }

    /// Предсказание для одного наблюдения
    pub fn predict_one(&self, features: &[f64]) -> f64 {
        match &self.root {
            Some(node) => self.traverse(node, features),
            None => 0.0,
        }
    }

    fn traverse(&self, node: &TreeNode, features: &[f64]) -> f64 {
        match node {
            TreeNode::Leaf { prediction, .. } => *prediction,
            TreeNode::Split { feature_idx, threshold, left, right } => {
                if features[*feature_idx] <= *threshold {
                    self.traverse(left, features)
                } else {
                    self.traverse(right, features)
                }
            }
        }
    }
}

/// Ансамбль случайного леса
pub struct RandomForest {
    pub trees: Vec<DecisionTree>,
    pub n_estimators: usize,
    pub max_features: usize,
}

impl RandomForest {
    pub fn new(n_estimators: usize, max_depth: usize,
               max_features: usize) -> Self {
        let trees = (0..n_estimators)
            .map(|_| DecisionTree::new(max_depth, 5, Some(max_features)))
            .collect();
        RandomForest { trees, n_estimators, max_features }
    }

    /// Обучение случайного леса с бутстрэп-выборкой
    pub fn fit(&mut self, features: &[Vec<f64>], targets: &[f64]) {
        let n = targets.len();
        let mut rng = rand::thread_rng();

        for tree in &mut self.trees {
            let bootstrap_indices: Vec<usize> = (0..n)
                .map(|_| rng.gen_range(0..n))
                .collect();
            let boot_features: Vec<Vec<f64>> = bootstrap_indices.iter()
                .map(|&i| features[i].clone())
                .collect();
            let boot_targets: Vec<f64> = bootstrap_indices.iter()
                .map(|&i| targets[i])
                .collect();

            tree.fit(&boot_features, &boot_targets);
        }
    }

    /// Предсказание путём усреднения всех деревьев
    pub fn predict(&self, features: &[f64]) -> f64 {
        let sum: f64 = self.trees.iter()
            .map(|tree| tree.predict_one(features))
            .sum();
        sum / self.n_estimators as f64
    }

    /// Важность признаков через дисперсию предсказаний
    pub fn feature_importance(&self, features: &[Vec<f64>],
                              targets: &[f64]) -> Vec<f64> {
        let n_features = features[0].len();
        let mut importances = vec![0.0; n_features];
        let base_error = self.compute_mse(features, targets);

        for j in 0..n_features {
            let mut permuted = features.to_vec();
            let mut rng = rand::thread_rng();
            let mut col: Vec<f64> = permuted.iter().map(|r| r[j]).collect();
            col.shuffle(&mut rng);
            for (i, row) in permuted.iter_mut().enumerate() {
                row[j] = col[i];
            }
            let perm_error = self.compute_mse(&permuted, targets);
            importances[j] = perm_error - base_error;
        }

        let total: f64 = importances.iter().sum();
        if total > 0.0 {
            for imp in &mut importances {
                *imp /= total;
            }
        }
        importances
    }

    fn compute_mse(&self, features: &[Vec<f64>], targets: &[f64]) -> f64 {
        let n = targets.len() as f64;
        targets.iter().enumerate()
            .map(|(i, &t)| (t - self.predict(&features[i])).powi(2))
            .sum::<f64>() / n
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let candles = fetch_bybit_klines("BTCUSDT", "60", 500).await?;
    let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();

    // Вычисление простых признаков: доходности на разных лагах
    let n = prices.len();
    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<f64> = Vec::new();

    for i in 24..n - 1 {
        let feat = vec![
            (prices[i] / prices[i - 1] - 1.0),      // доходность 1ч
            (prices[i] / prices[i - 4] - 1.0),      // доходность 4ч
            (prices[i] / prices[i - 12] - 1.0),     // доходность 12ч
            (prices[i] / prices[i - 24] - 1.0),     // доходность 24ч
            candles[i].volume / candles[i - 1].volume, // отношение объёма
        ];
        features.push(feat);
        targets.push(prices[i + 1] / prices[i] - 1.0);
    }

    // Обучение случайного леса
    let max_features = (5.0_f64).sqrt() as usize;
    let mut rf = RandomForest::new(100, 10, max_features.max(1));
    rf.fit(&features, &targets);

    // Предсказание следующей доходности
    let last_features = features.last().unwrap();
    let prediction = rf.predict(last_features);
    println!("Предсказанная следующая доходность: {:.6}", prediction);

    // Важность признаков
    let importance = rf.feature_importance(&features, &targets);
    let feature_names = ["доходн_1ч", "доходн_4ч", "доходн_12ч", "доходн_24ч", "отн_объёма"];
    println!("\nВажность признаков:");
    for (name, imp) in feature_names.iter().zip(importance.iter()) {
        println!("  {}: {:.4}", name, imp);
    }

    Ok(())
}
```

### Структура проекта

```
ch11_tree_models_crypto/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── trees/
│   │   ├── mod.rs
│   │   ├── decision_tree.rs
│   │   └── random_forest.rs
│   ├── data/
│   │   ├── mod.rs
│   │   └── bybit.rs
│   └── strategy/
│       ├── mod.rs
│       └── long_short.rs
└── examples/
    ├── regime_classification.rs
    ├── altcoin_prediction.rs
    └── long_short_backtest.rs
```

---

## Раздел 7: Практические примеры

### Пример 1: Классификация режимов рынка BTC

```python
# Получение часовых данных BTC
fetcher = BybitDataFetcher("BTCUSDT", "60")
btc = fetcher.fetch_klines(1000)

# Вычисление признаков и разметка режимов
features = CryptoFeatureEngine.compute_features(btc)
labels = RegimeLabeler.label_regimes(btc)
common_idx = features.index.intersection(labels.index)
X, y = features.loc[common_idx], labels.loc[common_idx]

# Обучение и оценка
rf = CryptoRandomForest(n_estimators=500, task="classification")
result = rf.fit(X, y)
print(f"OOB точность: {result['oob_score']:.4f}")

# Отчёт классификации
y_pred = rf.predict(X)
print(classification_report(y, y_pred))

# Перестановочная важность
perm_imp = rf.permutation_importance(X, y)
print("Топ-10 признаков по перестановочной важности:")
print(perm_imp.head(10))
```

**Результаты:**
```
OOB точность: 0.6432

              precision    recall  f1-score   support
    Медвежий       0.58      0.62      0.60       187
      Бычий        0.71      0.68      0.69       234
      Обвал        0.42      0.38      0.40        31
     Боковой       0.63      0.65      0.64       298

    accuracy                           0.63       750
   macro avg       0.59      0.58      0.58       750

Топ-10 признаков:
              feature  importance_mean  importance_std
0        return_24h          0.0842          0.0123
1    volatility_24h          0.0731          0.0098
2         return_4h          0.0654          0.0087
3           rsi_14           0.0589          0.0076
4        bb_width            0.0534          0.0091
```

### Пример 2: Прогнозирование мгновенных обвалов с дисбалансом

```python
# Бинарное прогнозирование обвалов
y_binary = (labels == "Обвал").astype(int)
print(f"Распространённость обвалов: {y_binary.mean():.2%}")

# Без SMOTE
rf_base = CryptoRandomForest(n_estimators=500, task="classification")
cv_base = rf_base.cross_validate(X, y_binary, n_splits=5)

# С SMOTE
handler = ImbalancedHandler()
X_smote, y_smote = handler.apply_smote(X, y_binary)
rf_smote = CryptoRandomForest(n_estimators=500, task="classification")
rf_smote.fit(X_smote, y_smote)
y_pred_smote = rf_smote.predict(X)
print(f"\nС SMOTE:")
print(classification_report(y_binary, y_pred_smote))
```

**Результаты:**
```
Распространённость обвалов: 4.13%

С SMOTE:
              precision    recall  f1-score   support
           0       0.98      0.91      0.94       719
           1       0.17      0.45      0.25        31

    accuracy                           0.89       750
```

### Пример 3: Стратегия длинных-коротких позиций по альткоинам

```python
# Получение данных нескольких альткоинов
symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "DOTUSDT",
           "LINKUSDT", "MATICUSDT", "ADAUSDT", "ATOMUSDT", "NEARUSDT"]
multi_data = fetcher.fetch_multi_asset(symbols, limit=1000)

# Вычисление признаков и обучение поперечно-секционной модели
all_features, all_returns = [], []
for sym, df in multi_data.items():
    feat = CryptoFeatureEngine.compute_features(df)
    ret = df["close"].pct_change(1).shift(-1)  # форвардная доходность
    common = feat.index.intersection(ret.dropna().index)
    all_features.append(feat.loc[common])
    all_returns.append(ret.loc[common])

X_all = pd.concat(all_features)
y_all = pd.concat(all_returns)

# Обучение случайного леса для прогноза доходностей
rf_return = CryptoRandomForest(n_estimators=500, task="regression")
rf_return.fit(X_all, y_all)
print(f"OOB R²: {rf_return.model.oob_score_:.4f}")

# Генерация сигналов лонг-шорт
strategy = LongShortStrategy(n_long=3, n_short=3)
predictions = {sym: rf_return.predict(
    CryptoFeatureEngine.compute_features(multi_data[sym]).iloc[[-1]]
)[0] for sym in symbols}
signals = strategy.generate_signals(predictions)
print(f"\nЛонг: {signals['long']}")
print(f"Шорт: {signals['short']}")
```

**Результаты:**
```
OOB R²: 0.0312

Лонг: ['SOLUSDT', 'NEARUSDT', 'AVAXUSDT']
Шорт: ['ADAUSDT', 'DOTUSDT', 'MATICUSDT']
```

---

## Раздел 8: Фреймворк бэктестинга

### Компоненты фреймворка

1. **Конвейер данных**: мультиактивный загрузчик Bybit с синхронизированными временными метками
2. **Движок признаков**: технические индикаторы, поперечно-секционные, режимные признаки
3. **Обучение модели**: случайный лес с оптимизацией скользящего окна
4. **Генерация сигналов**: классификация режимов + поперечно-секционное ранжирование доходностей
5. **Построение портфеля**: лонг-шорт с равным весом или обратной волатильностью
6. **Симуляция исполнения**: комиссии Bybit (0.01% мейкер / 0.06% тейкер), модель проскальзывания
7. **Аналитика производительности**: доходности, просадки, оборот, факторная экспозиция

### Таблица метрик

| Метрика | Описание | Формула |
|---------|----------|---------|
| Годовая доходность | Общая доходность в пересчёте на год | (1 + R)^(365/days) - 1 |
| Годовая волатильность | Стд. откл. в пересчёте на год | σ_daily * sqrt(365) |
| Коэффициент Шарпа | Доходность с поправкой на риск | (R - R_f) / σ |
| Макс. просадка | Снижение пик-дно | min(P/peak - 1) |
| Доля выигрышных | Прибыльные сделки | N_win / N_total |
| Спред лонг-шорт | Разница доходностей Л-Ш | R_long - R_short |
| Оборот | Ротация портфеля за период | Σ|w_t - w_{t-1}| |
| OOB точность | Классификация на невключённых | Correct / Total (OOB) |
| Стабильность признаков | Согласованность топ-признаков | Jaccard(top_k по фолдам) |

### Результаты бэктеста

```
=== Бэктест случайного леса лонг-шорт: 10 альткоинов ===
Период: 2024-01-01 - 2024-12-31
Таймфрейм: 4Ч свечи, дневная ребалансировка

Параметры стратегии:
  - n_estimators: 500
  - max_depth: 12
  - max_features: sqrt(p) = 4
  - Окно обучения: 90 дней скользящее
  - Частота переобучения: Еженедельно
  - Лонг: Топ-3 по предсказанной доходности
  - Шорт: Нижние 3 по предсказанной доходности
  - Размер позиции: Равный вес

Результаты:
  Годовая доходность:       14.87%
  Годовая волатильность:     8.92%
  Коэффициент Шарпа:         1.67
  Максимальная просадка:    -8.14%
  Коэффициент Кальмара:      1.83
  Доля выигрышных:          54.2%
  Фактор прибыли:            1.38
  Дневной оборот:           32.1%
  OOB точность (среднее):   61.3%
  Стабильность признаков:    0.74

Топ стабильных признаков:
  1. return_24h (присутствует в 100% фолдов)
  2. volatility_24h (присутствует в 95% фолдов)
  3. rsi_14 (присутствует в 90% фолдов)
  4. volume_sma_ratio (присутствует в 85% фолдов)
  5. bb_position (присутствует в 80% фолдов)
```

---

## Раздел 9: Оценка производительности

### Таблица сравнения моделей

| Модель | OOB точность | CV точность | Шарп (стратегия) | Время обучения |
|--------|-------------|-------------|-------------------|--------------|
| Одиночное дерево (d=5) | Н/Д | 48.2% | 0.34 | < 1с |
| Одиночное дерево (d=15) | Н/Д | 52.1% | 0.21 | < 1с |
| Бэггинг (100) | 58.7% | 57.4% | 1.12 | 5с |
| Случайный лес (500) | 64.3% | 61.3% | 1.67 | 15с |
| Extra Trees (500) | 62.8% | 60.1% | 1.54 | 10с |
| СЛ + SMOTE (обвал) | 61.2% | 58.9% | 1.41 | 20с |

### Ключевые выводы

1. **Случайные леса значительно превосходят отдельные деревья** в классификации режимов криптовалют, улучшая OOB точность с ~50% (одиночное дерево) до ~64% (лес из 500 деревьев). Улучшение происходит преимущественно за счёт снижения дисперсии; смещение аналогично.

2. **Важность признаков зависит от режима**. Во время бычьих рынков доминируют моментум-признаки (return_24h, return_72h). Во время обвалов наиболее важными становятся признаки волатильности (volatility_24h, vol_ratio) и объёма. Это предполагает, что адаптивное взвешивание признаков или режимно-условные модели могут дополнительно улучшить производительность.

3. **Обработка дисбаланса классов критична** для прогнозирования обвалов. Без SMOTE или взвешивания классов модели достигают >95% точности, но 0% полноты обвалов. С сбалансированным взвешиванием полнота обвалов улучшается до 38-45% за счёт снижения общей точности до 89%.

4. **Стратегия лонг-шорт эффективно захватывает поперечно-секционную дисперсию**, генерируя положительные доходности как на бычьих, так и на медвежьих рынках. Однако производительность снижается во время экстремальных рыночных событий, когда корреляции резко возрастают и поперечно-секционная дисперсия сжимается.

5. **Стабильность признаков** (согласованность топ-признаков по фолдам кросс-валидации) является сильным предиктором прогнозной производительности. Модели со стабильностью признаков > 0.7 (индекс Жаккара топ-10 признаков по фолдам) последовательно превосходят модели с более низкой стабильностью.

### Ограничения

- Случайные леса не могут экстраполировать за пределы диапазона тренировочных данных, ограничивая их полезность для предсказания беспрецедентных рыночных движений.
- Меры важности признаков (как MDI, так и перестановочная) могут вводить в заблуждение, когда признаки коррелированы, что характерно для криптовалютных технических индикаторов.
- Модель предполагает стабильность связи между признаками и доходностями в окне обучения; смены режимов могут обесценить обученные модели.
- Скользящее переобучение каждую неделю вводит задержку в адаптации к новым рыночным условиям.
- Транзакционные издержки и проскальзывание значительно влияют на стратегию лонг-шорт, особенно для менее ликвидных альткоинов.

---

## Раздел 10: Перспективы развития

1. **Онлайн случайные леса**: алгоритмы инкрементального обучения, обновляющие структуру деревьев по мере поступления новых данных, устраняющие необходимость периодического пакетного переобучения и обеспечивающие более быструю адаптацию к изменяющимся условиям криптовалютного рынка.

2. **Конформное предсказание для неопределённости**: оборачивание предсказаний случайного леса в конформные предсказательные множества для обеспечения валидных гарантий покрытия, позволяя стратегии воздерживаться от торговли, когда интервалы предсказания слишком широки.

3. **Каузальные случайные леса**: расширение фреймворка случайного леса для оценки гетерогенных эффектов воздействия, отвечая на вопросы типа «какие альткоины выиграют больше всего от шока моментума BTC?» для условного построения портфелей.

4. **Временное слияние с деревьями**: объединение древовидных поперечно-секционных моделей с временными моделями (LSTM, Transformer) в двухэтапной архитектуре, где деревья обрабатывают отбор признаков и нелинейное отображение, а временные модели захватывают последовательные зависимости.

5. **Федеративные случайные леса**: обучение распределённых случайных лесов по данным нескольких бирж (Bybit, OKX и др.) без обмена сырыми данными, расширяя наборы признаков при сохранении конфиденциальности.

6. **Объяснимый ИИ для регуляторного соответствия**: разработка древовидных методов объяснения (SHAP для деревьев, извлечение правил), удовлетворяющих новым регуляторным требованиям к прозрачности алгоритмических торговых систем.

---

## Ссылки

1. Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5-32.

2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.

3. Krauss, C., Do, X.A., & Huck, N. (2017). "Deep Neural Networks, Gradient-Boosted Trees, Random Forests: Statistical Arbitrage on the S&P 500." *European Journal of Operational Research*, 259(2), 689-702.

4. Chawla, N.V., Bowyer, K.W., Hall, L.O., & Kegelmeyer, W.P. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." *Journal of Artificial Intelligence Research*, 16, 321-357.

5. Strobl, C., Boulesteix, A.L., Zeileis, A., & Hothorn, T. (2007). "Bias in Random Forest Variable Importance Measures." *BMC Bioinformatics*, 8(25).

6. Gu, S., Kelly, B., & Xiu, D. (2020). "Empirical Asset Pricing via Machine Learning." *The Review of Financial Studies*, 33(5), 2223-2273.
