## Lacmus-server-test
Скрипт для тестирования новых inference-моделей, сравнения их точности и производительности. Предполагается, что ваша модель упакована в web-приложение, желательно docker container, как в [текущей реализации](https://github.com/lacmus-foundation/lacmus/blob/master/inference.py). Для работы в скрипте нужно указать путь к папке с PASCAL датасетом.  

#### Описание работы sync-test.py:  
1. Читаются *.xml аннотации из папки ```Annotations``` и конвертируются в COCO формат. Эта разметка используется как эталонная.  
2. Перебираются изображения из папки ```JPEGImages``` и по очереди отправляются в POST запросах на inference-сервер (по умолчанию на localhost:5000/image). 
 Время, затраченное на предикт каждого изображения, добавляется в словарь ```TIMES```   
3. Полученные результаты предиктов собираются в COCO-словарь и сравниваются с эталонной разметкой. Вычисляются стандартные метрики COCO и среднее время инференса. 

#### TODO:
Асинхронное тестирование с locust.