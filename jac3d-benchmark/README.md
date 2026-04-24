Jacobi 3D Benchmark

Файлы проекта:

    jac3d_cpu.c    # CPU версия (OpenMP)
    jac3d_cuda.cu  # GPU версия (CUDA)
    jac3d_common.h # Общие функции и заголовки
    verify.c       # Сравнение результатов CPU и GPU
    Makefile       # Сборка проекта
    read_bin.c     # Просмотр .bin файлов

Сборка:

    make all       # Собрать все исполняемые файлы
    make clean     # Удалить исполняемые файлы

Запуск:

    ./jac3d_cpu -L 384 -verify                   # CPU версия
    ./jac3d_cuda -L 384 -verify                  # GPU версия
    ./verify cpu_result.bin gpu_result.bin 384   # Сравнение результатов

Параметры:

    -L <size>      # Размер сетки (по умолчанию 384)
    -itmax <n>     # Максимум итераций (по умолчанию 20)
    -maxeps <val>  # Порог сходимости (по умолчанию 0.5)
    -verify        # Сохранить результат в .bin файл

Проверка:

    make test      # Запустить все тесты автоматически

Ожидаемый результат верификации: Verification: SUCCESSFUL
