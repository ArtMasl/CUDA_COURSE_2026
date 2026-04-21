ADI 3D Benchmark

Файлы проекта:

    adi3d_cpu.c    # CPU версия (OpenMP)
    adi3d_cuda.cu  # GPU версия (CUDA)
    Makefile       # Сборка проекта

Сборка:

    make all       # Собрать все исполняемые файлы
    make clean     # Удалить исполняемые файлы

Запуск:

    ./adi3d_cpu -L 384 -verify                   # CPU версия
    ./adi3d_cuda -L 384 -verify                  # GPU версия

Параметры:

    -L <size>      # Размер сетки (по умолчанию 384)
    -itmax <n>     # Максимум итераций (по умолчанию 10)

Проверка:

    make test      # Запустить все тесты автоматически
