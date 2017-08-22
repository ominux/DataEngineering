import dataset
import inspect

kaka = dataset.Dataset()
kaka.haha()

for name, obj in inspect.getmembers(dataset):
    print(name)
    print(obj)
    if inspect.isclass(obj):
        print(obj)
