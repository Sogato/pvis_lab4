import numpy
from mpi4py import MPI
import time
import itertools
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


def load_and_prepare_data(filepath):
    # Загрузка и подготовка данных.
    data = pd.read_csv(filepath)
    data['Date'] = pd.to_datetime(data['Date'])
    return data


def parallel_analyze_top_hashtags(args):
    # Функция для параллельного анализа. Принимает кортеж (data, timeframe).
    data, timeframe = args
    return analyze_top_hashtags(data, timeframe)


def analyze_top_hashtags(data, timeframe):
    # Анализ топ-10 хештегов для заданного временного промежутка.
    start_date, end_date = timeframe
    timeframe_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    hashtags = timeframe_data['Hashtags'].str.split(',').explode()
    return hashtags.value_counts().head(10)


def find_unique_hashtags(top_hashtags_by_timeframe):
    # Нахождение уникальных хештегов для каждого временного промежутка.
    return {timeframe: set(top_hashtags.index) - set(itertools.chain.from_iterable(
        [hashtags.index for tf, hashtags in top_hashtags_by_timeframe.items() if tf != timeframe]))
            for timeframe, top_hashtags in top_hashtags_by_timeframe.items()}


def compare_hashtag_distributions(top_hashtags_by_timeframe, timeframes):
    # Сравнение распределений хештегов между временными промежутками.
    def jaccard_distance(tf1, tf2):
        set1 = set(top_hashtags_by_timeframe[tf1].index)
        set2 = set(top_hashtags_by_timeframe[tf2].index)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return 1 - (intersection / union) if union != 0 else 1

    return {(tf1, tf2): jaccard_distance(tf1, tf2)
            for tf1, tf2 in itertools.combinations(timeframes, 2)}


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Определение временных промежутков ЧМ 2018
    timeframes = [
        ('2018-06-14', '2018-06-29'),  # Открытие чемпионата
        ('2018-06-30', '2018-07-03'),  # 1/8 финала
        ('2018-07-04', '2018-07-07'),  # 1/4 финала
        ('2018-07-08', '2018-07-14'),  # 1/2 финала
        ('2018-07-15', '2018-07-16'),  # Финал
    ]

    if rank == 0:
        data = load_and_prepare_data('FIFA.csv')
        split_data = numpy.array_split(data, size)
    else:
        split_data = None

    data_chunk = comm.scatter(split_data, root=0)
    for thread_count in [2, 4, 6, 8]:
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            results = list(executor.map(parallel_analyze_top_hashtags, [(data_chunk, tf) for tf in timeframes]))

        all_results = comm.gather(results, root=0)

        if rank == 0:
            # Объединение результатов от всех процессов
            combined_results = list(itertools.chain.from_iterable(all_results))

            # Структурирование данных для дальнейшего анализа
            top_hashtags_by_timeframe = {tf: pd.Series(dtype=object) for tf in timeframes}
            for result in combined_results:
                for tf, top_hashtags in result.items():
                    top_hashtags_by_timeframe[tf] = top_hashtags_by_timeframe[tf].add(top_hashtags, fill_value=0)

            # Анализ топ-10 хештегов для каждого временного промежутка
            print("\nТоп-10 хештегов для каждого временного промежутка:")
            for tf, top_hashtags in top_hashtags_by_timeframe.items():
                print(f"Промежуток {tf}:")
                print(top_hashtags, "\n")

            # Определение уникальности хештегов
            unique_hashtags = find_unique_hashtags(top_hashtags_by_timeframe)
            print("\nУникальные хештеги для каждого временного промежутка:")
            for tf, unique_ht in unique_hashtags.items():
                print(f"Промежуток {tf}:")
                print(unique_ht, "\n")

            # Сравнение распределений хештегов
            comparison_results = compare_hashtag_distributions(top_hashtags_by_timeframe, timeframes)
            closest_distributions = min(comparison_results, key=comparison_results.get)
            print("Два временных промежутка с наиболее близким распределением частоты хештегов:")
            for closest_distribution in closest_distributions:
                print(closest_distribution)

        execution_time = time.time() - start_time
        if rank == 0:
            print(f"{thread_count} потоков: {execution_time:.2f} секунд")
