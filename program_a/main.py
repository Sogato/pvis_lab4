import time
from mpi4py import MPI
from bs4 import BeautifulSoup
import requests
import re
from collections import Counter
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

# Инициализация MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

urls = [
    "https://ru.wikipedia.org/wiki/%D0%A2%D0%B8%D0%B3%D1%80",
    "https://ru.wikipedia.org/wiki/%D0%A3%D1%82%D0%BA%D0%BE%D0%BD%D0%BE%D1%81",
    "https://ru.wikipedia.org/wiki/%D0%9E%D0%B1%D1%8B%D0%BA%D0%BD%D0%BE%D0%B2%D0%B5%D0%BD%D0%BD%D0%B0%D1%8F_%D0%B1%D0%B5%D0%BB%D0%BA%D0%B0",
    "https://ru.wikipedia.org/wiki/%D0%9E%D0%B1%D1%8B%D0%BA%D0%BD%D0%BE%D0%B2%D0%B5%D0%BD%D0%BD%D1%8B%D0%B9_%D0%B1%D0%BE%D0%B1%D1%80",
    "https://ru.wikipedia.org/wiki/%D0%A1%D0%B5%D1%80%D0%B0%D1%8F_%D0%BA%D1%80%D1%8B%D1%81%D0%B0",
    "https://ru.wikipedia.org/wiki/%D0%93%D0%BE%D1%80%D0%B8%D0%BB%D0%BB%D1%8B",
]

# Распределение URL между процессами
urls_per_process = len(urls) // size
assigned_urls = urls[rank * urls_per_process:(rank + 1) * urls_per_process]


def parse_and_analyze(url):
    start_time = time.time()  # Начало замера времени для этого URL
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        for script in soup(["script", "style", "footer", "nav", "aside"]):
            script.extract()

        text = soup.get_text()
        words = re.findall(r'\b\w+\b', text.lower())
        filtered_words = [word for word in words if word not in stopwords.words('russian')]
        return Counter(filtered_words), time.time() - start_time  # Возвращаем результат и время
    except Exception as e:
        print(f"Ошибка при обработке {url}: {e}")
        return Counter(), time.time() - start_time


start_time = time.time()  # Начало замера времени для всего процесса

# Анализ каждого URL, назначенного процессу
word_counts = Counter()
processing_times = []  # Список для хранения времени обработки каждого URL
for url in assigned_urls:
    counts, time_taken = parse_and_analyze(url)
    word_counts += counts
    processing_times.append(time_taken)

# Сбор данных со всех процессов
all_counts = comm.gather(word_counts, root=0)
all_times = comm.gather(processing_times, root=0)

# Вывод результатов на главном процессе
if rank == 0:
    total_counts = Counter()
    total_time = 0
    for count, times in zip(all_counts, all_times):
        total_counts += count
        total_time += sum(times)  # Суммарное время обработки для каждого процесса

    most_common_words = total_counts.most_common(20)
    # Форматированный вывод результатов
    print(f"\nТест с N процессами:")
    print("Наиболее встречающиеся слова на всех страницах:")
    print("{:<3} {:<15} {:<10}".format("№", "Слово", "Количество"))
    for i, (word, count) in enumerate(most_common_words, 1):
        print("{:<3} {:<15} {:<10}".format(i, word, count))

    print(f"\nОбщее время обработки: {total_time:.2f} секунд")
    print(f"Время обработки каждым процессом: {[sum(times) for times in all_times]}")

total_process_time = time.time() - start_time  # Общее время выполнения текущего процесса
print(f"Время выполнения процесса {rank}: {total_process_time:.2f} секунд")
