import dataset
import inspect

for name, obj in inspect.getmembers(dataset):
    print(name)
    print(obj)
    if inspect.isclass(obj):
        print(obj)
