# ПВИС Лабораторная работа 4

---
Этот репозиторий содержит код для реализации распределенных многопроцессных и многопоточных 
вычислений в рамках четвертой лабораторной работы по предмету "Программирование вычислительных систем" (ПВИС), 
используя стандарт MPI.

---

## Требования
Перед запуском программ убедитесь, что у вас установлены Python, библиотеки для работы с MPI и другие необходимые 
зависимости, перечисленные в файле requirements.txt. Процесс установки следующий:
1. Клонируйте репозиторий:
```
git clone https://github.com/Sogato/pvis_lab4.git
```
2. Создайте виртуальное окружение:
```
python -m venv env
```
3. Активируйте виртуальное окружение:
```
source env/bin/activate  # Для Linux/macOS
env\Scripts\activate     # Для Windows
```
4. Установите зависимости проекта:
```
pip install -r requirements.txt
```

## Использование
### Program A
Для запуска программы А выполните:
```
mpiexec -n <число_процессов> python program_a/main.py
```
Программа A использует несколько процессов MPI для парсинга и анализа контента веб-страниц. 
Тестирование программы проводится с различным числом процессов, анализирующих различное количество веб-страниц.

### Program B
Для запуска программы B выполните:
```
mpiexec -n <число_процессов> python program_b/main.py
```
Программа B обрабатывает данные FIFA World Cup 2018 Tweets, используя многопроцессорную и многопоточную обработку 
с помощью MPI. Каждый процесс запускает одинаковое количество потоков для анализа своего фрагмента данных.
