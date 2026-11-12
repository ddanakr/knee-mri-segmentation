# Knee MRI Segmentation

Сегментация структур коленного сустава на МРТ с помощью 2D U-Net (MONAI) на датасете **OAIZIB-CM**.

Модель по каждому срезу МРТ предсказывает 6 классов:

| Метка | Класс |
|:-----:|-------|
| 0 | фон |
| 1 | бедренная кость (femur) |
| 2 | хрящ бедренной кости (femoral cartilage) |
| 3 | большеберцовая кость (tibia) |
| 4 | медиальный хрящ большеберцовой кости (medial tibial cartilage) |
| 5 | латеральный хрящ большеберцовой кости (lateral tibial cartilage) |

## Результаты

Метрики на тестовой выборке (`imagesTs` / `labelsTs`), чекпойнт `best.pth`. Dice и HD95 считаются по 2D-срезам, Hausdorff95 — в пикселях.

| Класс | Dice | Hausdorff95 |
|-------|:----:|:-----------:|
| femur | 0.974 | 3.11 |
| femoral cartilage | 0.873 | 3.79 |
| tibia | 0.970 | 2.39 |
| medial tibial cartilage | 0.812 | 3.73 |
| lateral tibial cartilage | 0.838 | 3.23 |
| **Mean Dice** | **0.894** | — |

Для сравнения, `last.pth` даёт mean Dice 0.882. Полные результаты лежат в `results/metrics/`, кривые обучения — в `results/figures/training_progress.png`.

## Структура проекта

```
knee-mri-segmentation/
├── configs/
│   ├── config.yaml          # основной конфиг (25 эпох, batch 16)
│   └── config_debug.yaml    # быстрый прогон для отладки (2 эпохи, batch 2)
├── notebooks/
│   └── train_knee_seg.ipynb # Colab-ноутбук: загрузка данных, препроцессинг, обучение, оценка
├── src/
│   ├── preprocess.py        # NIfTI -> нормализованные .npy
│   ├── dataset.py           # KneeDataset, индекс срезов, ChunkedVolumeSampler
│   ├── transforms.py        # аугментации
│   ├── model.py             # 2D U-Net (monai.networks.nets.UNet)
│   ├── train.py             # цикл обучения
│   ├── evaluate.py          # Dice и Hausdorff95 на тесте
│   ├── predict.py           # инференс на одном объёме + визуализация
│   └── utils.py             # конфиг, seed, чекпойнты, EarlyStopping, графики
└── requirements.txt
```

Каталоги `data/`, `checkpoints/`, `results/`, `logs/` добавлены в `.gitignore`.

## Как это работает

- **Данные.** Датасет [OAIZIB-CM](https://huggingface.co/datasets/YongchengYAO/OAIZIB-CM) скачивается с Hugging Face. Ожидаемая структура: `imagesTr/`, `labelsTr/`, `imagesTs/`, `labelsTs/`. Файлы изображений называются `<case>_0000.nii.gz`, маски — `<case>.nii.gz`.
- **Препроцессинг.** Каждый объём нормализуется по ненулевым вокселям (z-score) и сохраняется как `.npy`, чтобы читать срезы через `mmap` без повторной загрузки NIfTI.
- **Обучение 2D.** Объём режется на срезы вдоль `axis` (по умолчанию 0), каждый срез ресайзится до `target_size` (256×256; изображение билинейно, маска — nearest).
- **Модель.** MONAI `UNet`, 2D, каналы `[32, 64, 128, 256, 512]`, 2 residual-блока на уровень, instance norm, dropout 0.1.
- **Функция потерь и оптимизация.** `DiceCELoss` + Adam, `ReduceLROnPlateau` по val Dice, early stopping (patience 5).
- **Аугментации.** Случайные отражения, повороты на 90°, сдвиг интенсивности.
- **Сэмплер.** Весь препроцессированный датасет (~40 GB) не помещается в RAM, поэтому `ChunkedVolumeSampler` на каждой эпохе перемешивает порядок объёмов и срезы внутри группы из `shuffle_chunk_volumes` объёмов. Так читается меньше «холодных» файлов, а перемешивание остаётся достаточным.
- **Чекпойнты.** `last.pth` (модель + оптимизатор) сохраняется после каждой эпохи и каждые `checkpoint_every_n_steps` шагов; `best.pth` — при улучшении val Dice. Обучение можно продолжить с `resume=True`.

## Установка

```bash
pip install -r requirements.txt
```

Нужен GPU с CUDA для нормального времени обучения (на CPU код тоже работает, но очень медленно).

## Запуск

Проект писался под **Google Colab** с Google Drive: пути в конфигах и в блоках `if __name__ == "__main__"` указывают на `/content/...`. Перед запуском локально поправьте пути в `configs/config.yaml` (`data_dir`, `processed_dir`, `checkpoints_dir`, `logs_dir`, `results_dir`).

1. **Скачать данные**

   ```python
   from huggingface_hub import snapshot_download

   snapshot_download(
       repo_id="YongchengYAO/OAIZIB-CM",
       repo_type="dataset",
       local_dir="data/raw",
   )
   ```

2. **Препроцессинг**

   ```python
   from src.preprocess import main
   main("configs/config.yaml")
   ```

3. **Обучение** (быстрая проверка — с `config_debug.yaml`)

   ```python
   from src.train import main
   main("configs/config.yaml")               # с нуля
   main("configs/config.yaml", resume=True)  # продолжить с last.pth
   ```

4. **Оценка на тесте**

   ```python
   from src.evaluate import evaluate
   evaluate("configs/config.yaml")                                    # best.pth
   evaluate("configs/config.yaml", f"{ckpt_dir}/last.pth", "test_evaluation_last")
   ```

5. **Предсказание для одного объёма**

   ```python
   from src.predict import main
   main(
       config_path="configs/config.yaml",
       image_path="data/raw/imagesTs/oaizib_406_0000.nii.gz",
       mask_path="data/raw/labelsTs/oaizib_406.nii.gz",  # необязательно
       slice_idx=None,                                   # по умолчанию — средний срез
   )
   ```

   Результат (МРТ / разметка / предсказание) сохраняется в `<results_dir>/figures/`.

В Colab удобнее всего пройти все шаги по порядку в [notebooks/train_knee_seg.ipynb](notebooks/train_knee_seg.ipynb).

## Основные параметры конфига

| Параметр | Описание |
|----------|----------|
| `axis` | ось объёма, по которой берутся срезы |
| `target_size` | размер среза после ресайза |
| `val_split` | доля объёмов для валидации (разбиение по объёмам, не по срезам) |
| `batch_size`, `learning_rate`, `epochs` | параметры обучения |
| `scheduler_patience` | терпение `ReduceLROnPlateau` |
| `early_stopping_patience`, `early_stopping_mode` | ранняя остановка |
| `checkpoint_every_n_steps` | как часто сохранять `last.pth` внутри эпохи |
| `shuffle_chunk_volumes` | размер группы объёмов в `ChunkedVolumeSampler` |
| `in_channels`, `out_channels`, `channels`, `strides`, `num_res_units`, `norm`, `dropout` | архитектура U-Net |

## Ограничения

- Модель 2D: контекст между соседними срезами не используется, метрики считаются по срезам, а не по 3D-объёмам.
- Самые сложные классы — хрящи большеберцовой кости (Dice ≈ 0.81–0.84).
- Датасет и веса в репозиторий не входят.
